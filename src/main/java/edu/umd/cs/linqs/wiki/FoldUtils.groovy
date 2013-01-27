package edu.umd.cs.linqs.wiki

import edu.emory.mathcs.jplasma.tdouble.DbdlConvert;
import edu.umd.cs.psl.database.DataStore
import edu.umd.cs.psl.database.Database
import edu.umd.cs.psl.database.DatabasePopulator
import edu.umd.cs.psl.database.DatabaseQuery
import edu.umd.cs.psl.database.Partition
import edu.umd.cs.psl.database.ResultList
import edu.umd.cs.psl.database.loading.Inserter;
import edu.umd.cs.psl.database.loading.Updater
import edu.umd.cs.psl.model.argument.GroundTerm
import edu.umd.cs.psl.model.argument.Variable
import edu.umd.cs.psl.model.atom.Atom
import edu.umd.cs.psl.model.atom.GroundAtom
import edu.umd.cs.psl.model.atom.QueryAtom
import edu.umd.cs.psl.model.predicate.Predicate
import edu.umd.cs.psl.model.predicate.StandardPredicate


import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

class FoldUtils {


	private static Logger log = LoggerFactory.getLogger("FoldUtils");

	/**
	 * creates two splits of data by randomly sampling from partition fullData 
	 * and places one split in partition train and one in partition test.
	 * @param data
	 * @param ratio
	 * @param fullData
	 * @param train
	 * @param test
	 */
	static Set<GroundTerm> [] generateRandomSplit(DataStore data, double ratio,
			Partition observedData, Partition groundTruth, Partition train,
			Partition test, Partition trainLabels, Set<DatabaseQuery> queries,
			Set<Variable> keys) {
		Random rand = new Random()

		log.debug("Splitting data from " + observedData + " with ratio " + ratio +
				" into new partitions " + train +" and " + test)

		def predicates = data.getRegisteredPredicates()
		Database db = data.getDatabase(observedData)
		Map<GroundTerm, Partition> keyMap = new HashMap<GroundTerm, Partition>()
		for (DatabaseQuery q : queries) {
			ResultList groundings = db.executeQuery(q)

			for (Variable key : keys) {
				int keyIndex = q.getVariableIndex(key)
				if (keyIndex == -1)
					continue
				for (int i = 0; i < groundings.size(); i++) {
					GroundTerm [] grounding = groundings.get(i)
					Partition p = (rand.nextDouble() < ratio) ? train : test;
					keyMap.put(grounding[keyIndex], p)
				}
			}
		}

		def splits = new HashSet<GroundTerm>[2]
		splits[0] = new HashSet<GroundTerm>()
		splits[1] = new HashSet<GroundTerm>()
		for (Map.Entry<GroundTerm, Partition> e : keyMap.entrySet()) {
			int index = (e.getValue() == train) ? 0 : 1
			splits[index].add(e.getKey())
		}

		log.debug("Assigned " + splits[0].size() + " in train partition and " + splits[1].size() + " in test")
		//log.debug("Found " + keyMap.size() + " primary keys.")

		for (Partition p : [train, test]) {
			log.debug("Putting data into partition " + p)
			for (DatabaseQuery q : queries) {
				// get predicate from query
				Predicate predicate = getPredicate(q)

				Inserter insert = data.getInserter(predicate, p)

				ResultList groundings = db.executeQuery(q)
				for (int i = 0; i < groundings.size(); i++) {
					GroundTerm [] grounding = groundings.get(i)
					// check if all keys in this ground term are in this split
					boolean add = true
					for (Variable key : keys) {
						int keyIndex = q.getVariableIndex(key)
						if (keyIndex != -1 && keyMap.get(grounding[keyIndex]) != p)
							add = false
					}
					if (add) {
						GroundAtom groundAtom = db.getAtom(predicate,  grounding)
						insert.insertValue(groundAtom.getValue(), groundAtom.getArguments())
					}
				}
			}
		}

		// move training labels from groundTruth into trainLabels
		log.debug("Moving ground truth into split training label partitions")
		for (DatabaseQuery q : queries) {
			Predicate predicate = getPredicate(q)

			Inserter insert = data.getInserter(predicate, trainLabels)

			ResultList groundings = db.executeQuery(q)
			for (int i = 0; i < groundings.size(); i++) {
				GroundTerm [] grounding = groundings.get(i)
				// check if all keys in this ground term are in this split
				boolean add = true
				for (Variable key : keys) {
					int keyIndex = q.getVariableIndex(key)
					if (keyIndex != -1 && keyMap.get(grounding[keyIndex]) != train)
						add = false
				}
				if (add) {
					GroundAtom groundAtom = db.getAtom(predicate,  grounding)
					insert.insertValue(groundAtom.getValue(), groundAtom.getArguments())
				}
			}
		}

		db.close()
		return splits
	}

	private static Predicate getPredicate(DatabaseQuery q) {
		Set<Atom> atoms = new HashSet<Atom>()
		q.getFormula().getAtoms(atoms)
		if (atoms.size() > 1)
			throw new IllegalArgumentException("Fold splitting query must be a single atom")
		Atom atom = atoms.iterator().next()
		Predicate predicate = atom.getPredicate()
		return predicate
	}
}