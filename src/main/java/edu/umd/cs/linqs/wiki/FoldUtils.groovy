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
	 * @param trainTestRatio
	 * @param fullData
	 * @param train
	 * @param test
	 * @param filterRatio
	 */
	static Set<GroundTerm> [] generateRandomSplit(DataStore data, double trainTestRatio,
			Partition observedData, Partition groundTruth, Partition train,
			Partition test, Partition trainLabels, Partition testLabels, Set<DatabaseQuery> queries,
			Set<Variable> keys, double filterRatio) {
		Random rand = new Random(0)

		log.debug("Splitting data from " + observedData + " with ratio " + trainTestRatio +
				" into new partitions " + train +" and " + test)

		Partition dummy = new Partition(99999);

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
					Partition p = (rand.nextDouble() < trainTestRatio) ? train : test;
					if (rand.nextDouble() > filterRatio) p = dummy;
					keyMap.put(grounding[keyIndex], p)
				}
			}
		}
		log.debug("Found {} unique keys", keyMap.size());

		def splits = new HashSet<GroundTerm>[2]
		splits[0] = new HashSet<GroundTerm>()
		splits[1] = new HashSet<GroundTerm>()
		for (Map.Entry<GroundTerm, Partition> e : keyMap.entrySet()) {
			int index = -1;
			if (e.getValue() == train) index = 0;
			if (e.getValue() == test) index = 1;
			if (index >= 0)
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
						log.trace("Inserted " + groundAtom + " into " + p)
					}
				}
			}
		}

		db.close()

		db = data.getDatabase(groundTruth)

		// move labels from groundTruth into trainLabels and testLabels
		log.debug("Moving ground truth into split training and testing label partitions")
		for (DatabaseQuery q : queries) {
			Predicate predicate = getPredicate(q)
			// insert into train label partition
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
					log.trace("Inserted " + groundAtom + " into " + trainLabels)
				}
			}

			insert = data.getInserter(predicate, testLabels)

			groundings = db.executeQuery(q)
			for (int i = 0; i < groundings.size(); i++) {
				GroundTerm [] grounding = groundings.get(i)
				// check if all keys in this ground term are in this split
				boolean add = true
				for (Variable key : keys) {
					int keyIndex = q.getVariableIndex(key)
					if (keyIndex != -1 && keyMap.get(grounding[keyIndex]) != test)
						add = false
				}
				if (add) {
					GroundAtom groundAtom = db.getAtom(predicate,  grounding)
					insert.insertValue(groundAtom.getValue(), groundAtom.getArguments())
					log.trace("Inserted " + groundAtom + " into " + testLabels)
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