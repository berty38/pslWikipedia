package edu.umd.cs.linqs.wiki

import java.util.Set;

import org.slf4j.Logger
import org.slf4j.LoggerFactory

import edu.emory.mathcs.utils.ConcurrencyUtils
import edu.umd.cs.psl.application.inference.MPEInference
import edu.umd.cs.psl.application.learning.weight.MaxLikelihoodMPE
import edu.umd.cs.psl.application.learning.weight.VotedPerceptron
import edu.umd.cs.psl.config.*
import edu.umd.cs.psl.core.*
import edu.umd.cs.psl.core.inference.*
import edu.umd.cs.psl.database.DataStore
import edu.umd.cs.psl.database.Database
import edu.umd.cs.psl.database.DatabasePopulator
import edu.umd.cs.psl.database.DatabaseQuery
import edu.umd.cs.psl.database.Partition
import edu.umd.cs.psl.database.ResultList
import edu.umd.cs.psl.database.rdbms.RDBMSDataStore
import edu.umd.cs.psl.database.rdbms.driver.H2DatabaseDriver
import edu.umd.cs.psl.database.rdbms.driver.H2DatabaseDriver.Type
import edu.umd.cs.psl.evaluation.result.*
import edu.umd.cs.psl.evaluation.statistics.DiscretePredictionComparator
import edu.umd.cs.psl.evaluation.statistics.DiscretePredictionStatistics
import edu.umd.cs.psl.evaluation.statistics.filter.MaxValueFilter
import edu.umd.cs.psl.groovy.*
import edu.umd.cs.psl.model.Model;
import edu.umd.cs.psl.model.argument.ArgumentType
import edu.umd.cs.psl.model.argument.GroundTerm
import edu.umd.cs.psl.model.argument.Variable
import edu.umd.cs.psl.model.atom.GroundAtom
import edu.umd.cs.psl.model.atom.QueryAtom
import edu.umd.cs.psl.model.function.AttributeSimilarityFunction
import edu.umd.cs.psl.ui.loading.*
import edu.umd.cs.psl.util.database.Queries

Logger log = LoggerFactory.getLogger(this.class)

ConfigManager cm = ConfigManager.getManager()
ConfigBundle wikiBundle = cm.getBundle("wiki")

def defaultPath = System.getProperty("java.io.tmpdir")
String dbpath = wikiBundle.getString("dbpath", defaultPath + "psl")
DataStore data = new RDBMSDataStore(new H2DatabaseDriver(Type.Disk, dbpath, true), wikiBundle)

PSLModel m = new PSLModel(this, data)

m.add predicate: "ClassifyCat", types: [ArgumentType.UniqueID, ArgumentType.UniqueID]
m.add predicate: "HasCat", types: [ArgumentType.UniqueID, ArgumentType.UniqueID]
m.add predicate: "Link", types: [ArgumentType.UniqueID, ArgumentType.UniqueID]
m.add predicate: "Talk", types: [ArgumentType.UniqueID, ArgumentType.UniqueID]

m.add rule : ( ClassifyCat(A,N) ) >> HasCat(A,N),  weight : 1.0
m.add rule : ( HasCat(B,C) & Link(A,B)) >> HasCat(A,C), weight: 1.0
m.add rule : ( Talk(D,A) & Talk(E,A) & HasCat(E,C) & (E - D) ) >> HasCat(D,C), weight: 1.0

m.add PredicateConstraint.Functional , on : HasCat

Partition fullObserved =  new Partition(0)
Partition groundTruth = new Partition(1)

def dataPath = "./data/"
def inserter
inserter = data.getInserter(Link, fullObserved)
InserterUtils.loadDelimitedData(inserter, dataPath + "uniqueLinks.txt")
inserter = data.getInserter(Talk, fullObserved)
InserterUtils.loadDelimitedData(inserter, dataPath + "talk.txt")
inserter = data.getInserter(HasCat, groundTruth)
InserterUtils.loadDelimitedData(inserter, dataPath + "newCategoryBelonging.txt")

folds = 1
trainReadPartitions = new ArrayList<Partition>()
testReadPartitions = new ArrayList<Partition>()
trainWritePartitions = new ArrayList<Partition>()
testWritePartitions = new ArrayList<Partition>()
trainLabelPartitions = new ArrayList<Partition>()

def keys = new HashSet<Variable>()
Variable document = new Variable("Document")
Variable linkedDocument = new Variable("LinkedDoc")
ArrayList<Set<Integer>> trainingKeys = new ArrayList<Set<Integer>>()
ArrayList<Set<Integer>> foldKeys = new ArrayList<Set<Integer>>()
def queries = new HashSet<DatabaseQuery>()
queries.add(new DatabaseQuery(ClassifyCat(document,N).getFormula()))
queries.add(new DatabaseQuery(Link(document, linkedDocument).getFormula()))
queries.add(new DatabaseQuery(Talk(document, A).getFormula()))
queries.add(new DatabaseQuery(HasCat(document, A).getFormula()))
keys.add(document)
keys.add(linkedDocument)
def partitionDocuments = new HashMap<Partition, Set<GroundTerm>>()

Random rand = new Random(1) // TODO: after debugging, remove fixed seed
double nbTrainingRatio = 0.3

for (int i = 0; i < folds; i++) {
	trainReadPartitions.add(i, new Partition(i + 2))
	testReadPartitions.add(i, new Partition(i + folds + 2))

	trainWritePartitions.add(i, new Partition(i + 2*folds + 2))
	testWritePartitions.add(i, new Partition(i + 3*folds + 2))

	trainLabelPartitions.add(i, new Partition(i + 4*folds + 2))

	Set<GroundTerm> [] documents = FoldUtils.generateRandomSplit(data, 0.5,
			fullObserved, groundTruth, trainReadPartitions.get(i),
			testReadPartitions.get(i), trainLabelPartitions.get(i), queries,
			keys)
	partitionDocuments.put(trainReadPartitions.get(i), documents[0])
	partitionDocuments.put(testReadPartitions.get(i), documents[1])

	trainingKeys.add(i, new HashSet<Integer>())
	foldKeys.add(i, new HashSet<Integer>())

	for (GroundTerm doc : partitionDocuments.get(trainReadPartitions.get(i))) {
		if (rand.nextDouble() < nbTrainingRatio)
			trainingKeys.get(i).add(Integer.decode(doc.toString()))
		foldKeys.get(i).add(Integer.decode(doc.toString()))
	}

	// add all trainingKeys into observed partition
	Database db = data.getDatabase(trainLabelPartitions.get(i))
	inserter = data.getInserter(HasCat, trainReadPartitions.get(i))
	ResultList res = db.executeQuery(new DatabaseQuery(HasCat(X,Y).getFormula()))
	for (GroundAtom atom : Queries.getAllAtoms(db, HasCat)) {
		Integer atomKey = Integer.decode(atom.getArguments()[0].toString())
		if (trainingKeys.get(i).contains(atomKey)) {
			inserter.insertValue(atom.getValue(), atom.getArguments())
			//log.debug("inserting " + atom.toString() + " w/ value " + atom.getValue())
		}
	}
	db.close()
}

for (int fold = 0; fold < folds; fold++) {
	Partition observedPartition = trainReadPartitions.get(fold)

	// do Naive Bayes training
	NaiveBayesUtil nb = new NaiveBayesUtil()

	log.debug("training set unique IDS: " + trainingKeys)
	nb.learn(trainingKeys.get(fold), "data/newCategoryBelonging.txt", "data/pruned-document.txt")

	inserter = data.getInserter(ClassifyCat, observedPartition)
	nb.insertAllProbabilities("data/pruned-document.txt", foldKeys.get(fold), inserter)

	def numCategories = 20

	Database labelsDB = data.getDatabase(trainLabelPartitions.get(fold), [HasCat] as Set)
	Database db = data.getDatabase(trainWritePartitions.get(fold), observedPartition)
	DatabasePopulator dbPop = new DatabasePopulator(db)
	Variable Category = new Variable("Category")
	Set<GroundTerm> categoryGroundings = new HashSet<GroundTerm>()
	for (int i = 0; i < numCategories; i++)
		categoryGroundings.add(data.getUniqueID(i))


	Variable Document = new Variable("Document")
	Map<Variable, Set<GroundTerm>> substitutions = new HashMap<Variable, Set<GroundTerm>>()
	substitutions.put(Document, partitionDocuments.get(observedPartition))
	substitutions.put(Category, categoryGroundings)
	dbPop.populate(new QueryAtom(HasCat, Document, Category), substitutions)

	/*
	 * Weight learning
	 */
	//	MaxLikelihoodMPE mle = new MaxLikelihoodMPE(m, db, labelsDB, wikiBundle)
	//	mle.learn()
	//	log.debug(m.toString())

	/*
	 * Inference
	 */
	MPEInference mpe = new MPEInference(m, db, wikiBundle)
	FullInferenceResult result = mpe.mpeInference()
	System.out.println("Objective: " + result.getTotalIncompatibility())

	/*
	 * Evaluation
	 */
	def comparator = new DiscretePredictionComparator(db)
	def groundTruthDB = data.getDatabase(groundTruth, [HasCat] as Set)
	comparator.setBaseline(groundTruthDB)
	comparator.setResultFilter(new MaxValueFilter(HasCat, 1))
	comparator.setThreshold(Double.MIN_VALUE) // treat best nonzero value as true

	DiscretePredictionStatistics stats = comparator.compare(HasCat)
	System.out.println("F1 score " + stats.getF1(
			DiscretePredictionStatistics.BinaryClass.POSITIVE))

	db.close()
	groundTruthDB.close()
}