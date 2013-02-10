package edu.umd.cs.linqs.wiki

import java.lang.annotation.Documented;
import java.util.Set;

import junit.framework.TestResult;

import org.eclipse.jdt.internal.core.util.LRUCache.Stats;
import org.slf4j.Logger
import org.slf4j.LoggerFactory

import edu.emory.mathcs.utils.ConcurrencyUtils
import edu.umd.cs.psl.application.inference.MPEInference
import edu.umd.cs.psl.application.learning.weight.maxlikelihood.MaxLikelihoodMPE
import edu.umd.cs.psl.application.learning.weight.maxmargin.MaxMargin
import edu.umd.cs.psl.application.learning.weight.maxlikelihood.MaxPseudoLikelihood
import edu.umd.cs.psl.application.learning.weight.random.FirstOrderMetropolisRandOM
import edu.umd.cs.psl.application.learning.weight.random.HardEMRandOM
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
import edu.umd.cs.psl.evaluation.statistics.RankingScore
import edu.umd.cs.psl.evaluation.statistics.SimpleRankingComparator
import edu.umd.cs.psl.evaluation.statistics.filter.MaxValueFilter
import edu.umd.cs.psl.groovy.*
import edu.umd.cs.psl.model.Model;
import edu.umd.cs.psl.model.argument.ArgumentType
import edu.umd.cs.psl.model.argument.GroundTerm
import edu.umd.cs.psl.model.argument.UniqueID
import edu.umd.cs.psl.model.argument.Variable
import edu.umd.cs.psl.model.atom.GroundAtom
import edu.umd.cs.psl.model.atom.QueryAtom
import edu.umd.cs.psl.model.atom.RandomVariableAtom;
import edu.umd.cs.psl.model.function.AttributeSimilarityFunction
import edu.umd.cs.psl.ui.loading.*
import edu.umd.cs.psl.util.database.Queries
import edu.umd.cs.psl.model.kernel.CompatibilityKernel;
import edu.umd.cs.psl.model.parameters.PositiveWeight;
import edu.umd.cs.psl.model.parameters.Weight
import edu.umd.cs.psl.model.predicate.Predicate;

import com.google.common.collect.Iterables;

//methods = ["RandOM", "MM1", "MM10", "MM100", "MM1000", "MLE"]
//methods = ["MM1", "MM10", "MM100", "MM1000", "MLE"]
methods = ["None"]
//methods = ["MPLE", "NONE", "MLE", "MM1", "MM10"]
//methods = ["MM1", "NB"]


/**
 * CONFIGURATION PARAMETERS
 */

dataPath = "./scraper/"
numCategories = 29
wordFile = "document.txt"
labelFile = "labels.txt"
linkFile = "links.txt"
talkFile = "talk.txt"
//dataPath = "./toydata/"
//numCategories = 4
//wordFile = "document.txt"
//labelFile = "labels.txt"
//linkFile = "links.txt"
//talkFile = "talk.txt"
//dataPath = "./data/"
//numCategories = 19
//wordFile = "pruned-document.txt"
//labelFile = "newCategoryBelonging.txt"
//linkFile = "uniqueLinks.txt"
//talkFile = "talk.txt"
sq = false
Random rand = new Random(0)
double trainingObservedRatio = 0.5 // ratio of training set for training NB
folds = 1 // number of folds
trainTestRatio = 0.5 // ratio of train to test splits (random)
filterRatio = 1.0 // ratio of documents to keep (throw away the rest)
targetSize = 3000 // target size of snowball sampler
explore = 0.001 // prob of random node in snowball sampler


Logger log = LoggerFactory.getLogger(this.class)

ConfigManager cm = ConfigManager.getManager()
ConfigBundle wikiBundle = cm.getBundle("wiki")

def defaultPath = System.getProperty("java.io.tmpdir")
String dbpath = wikiBundle.getString("dbpath", defaultPath + File.separator + "psl")
DataStore data = new RDBMSDataStore(new H2DatabaseDriver(Type.Disk, dbpath, true), wikiBundle)

PSLModel m = new PSLModel(this, data)

/*
 * DEFINE MODEL
 */
m.add predicate: "ClassifyCat", types: [ArgumentType.UniqueID, ArgumentType.UniqueID]
m.add predicate: "HasCat", types: [ArgumentType.UniqueID, ArgumentType.UniqueID]
m.add predicate: "Link", types: [ArgumentType.UniqueID, ArgumentType.UniqueID]
m.add predicate: "Cat", types: [ArgumentType.UniqueID]
m.add setcomparison: "avgValue", on: HasCat, using: SetComparison.Average
//m.add predicate: "Talk", types: [ArgumentType.UniqueID, ArgumentType.UniqueID]

//prior
m.add rule : ~(HasCat(A,N)), weight: 0.001, squared: sq

m.add rule : ( ClassifyCat(D,C) ) >> HasCat(D,C),  weight : 1.0, squared: sq
//m.add rule : ( HasCat(D,C) ) >> ClassifyCat(D,C), weight : 1.0, squared : sq
m.add rule : ( avgValue({D.Link}, {L})) >> HasCat(D,L), weight : 1.0, squared: sq
//m.add rule : ( HasCat(D,C) ) >> (avgValue({D.Link}, {C})), weight : 1.0, squared: sq
//m.add rule : ( HasCat(A,C) & Link(A,B) & (A - B)) >> HasCat(B,C), weight: 0.0, squared: sq
//m.add rule : ( Talk(D,A) & Talk(E,A) & HasCat(E,C) & (E - D) & (E ^ D) ) >> HasCat(D,C), weight: 1.0, squared: sq
//for (int i = 0; i < numCategories; i++)  {
//	UniqueID cat = data.getUniqueID(i)
//m.add rule : ( ClassifyCat(A,cat) ) >> HasCat(A,cat),  weight : 1.0, squared: sq
//	m.add rule : ( HasCat(B, cat) & Link(A,B)) >> HasCat(A, cat), weight: 0.0, squared: sq
//m.add rule : ( Talk(D,A) & Talk(E,A) & HasCat(E,cat) & (E - D) ) >> HasCat(D,cat), weight: 1.0, squared: sq
//}

m.add PredicateConstraint.PartialFunctional , on : HasCat


Partition fullObserved =  new Partition(0)
Partition groundTruth = new Partition(1)



/*
 * LOAD DATA
 */
def inserter
inserter = data.getInserter(Link, fullObserved)
InserterUtils.loadDelimitedData(inserter, dataPath + linkFile)
//inserter = data.getInserter(Talk, fullObserved)
//InserterUtils.loadDelimitedData(inserter, dataPath + talkFile)
inserter = data.getInserter(HasCat, groundTruth)
InserterUtils.loadDelimitedData(inserter, dataPath + labelFile)

trainReadPartitions = new ArrayList<Partition>()
testReadPartitions = new ArrayList<Partition>()
trainWritePartitions = new ArrayList<Partition>()
testWritePartitions = new ArrayList<Partition>()
trainLabelPartitions = new ArrayList<Partition>()
testLabelPartitions = new ArrayList<Partition>()

def keys = new HashSet<Variable>()
ArrayList<Set<Integer>> nbTrainingKeys = new ArrayList<Set<Integer>>()
ArrayList<Set<Integer>> trainingKeys = new ArrayList<Set<Integer>>()
ArrayList<Set<Integer>> testingKeys = new ArrayList<Set<Integer>>()
def queries = new HashSet<DatabaseQuery>()


/*
 * DEFINE PRIMARY KEY QUERIES FOR FOLD SPLITTING
 */
Variable document = new Variable("Document")
Variable linkedDocument = new Variable("LinkedDoc")
keys.add(document)
keys.add(linkedDocument)
//queries.add(new DatabaseQuery(ClassifyCat(document,N).getFormula()))
queries.add(new DatabaseQuery(Link(document, linkedDocument).getFormula()))
//queries.add(new DatabaseQuery(Talk(document, A).getFormula()))
queries.add(new DatabaseQuery(HasCat(document, A).getFormula()))

def partitionDocuments = new HashMap<Partition, Set<GroundTerm>>()

for (int i = 0; i < folds; i++) {
	trainReadPartitions.add(i, new Partition(i + 2))
	testReadPartitions.add(i, new Partition(i + folds + 2))

	trainWritePartitions.add(i, new Partition(i + 2*folds + 2))
	testWritePartitions.add(i, new Partition(i + 3*folds + 2))

	trainLabelPartitions.add(i, new Partition(i + 4*folds + 2))
	testLabelPartitions.add(i, new Partition(i + 5*folds + 2))

	Set<GroundTerm> [] documents = FoldUtils.generateRandomSplit(data, trainTestRatio,
			fullObserved, groundTruth, trainReadPartitions.get(i),
			testReadPartitions.get(i), trainLabelPartitions.get(i),
			testLabelPartitions.get(i), queries, keys, filterRatio)
	//	Set<GroundTerm> [] documents = FoldUtils.generateSnowballSplit(data, fullObserved, groundTruth,
	//		trainReadPartitions.get(i), testReadPartitions.get(i), trainLabelPartitions.get(i),
	//		testLabelPartitions.get(i), queries, keys, targetSize, Link, explore)




	partitionDocuments.put(trainReadPartitions.get(i), documents[0])
	partitionDocuments.put(testReadPartitions.get(i), documents[1])

	nbTrainingKeys.add(i, new HashSet<Integer>())
	trainingKeys.add(i, new HashSet<Integer>())
	testingKeys.add(i, new HashSet<Integer>())

	for (GroundTerm doc : partitionDocuments.get(trainReadPartitions.get(i))) {
		if (rand.nextDouble() < trainingObservedRatio)
			nbTrainingKeys.get(i).add(Integer.decode(doc.toString()))
		trainingKeys.get(i).add(Integer.decode(doc.toString()))
	}
	for (GroundTerm doc : partitionDocuments.get(testReadPartitions.get(i))) {
		testingKeys.get(i).add(Integer.decode(doc.toString()))
	}
}

Map<String, List<DiscretePredictionStatistics>> results = new HashMap<String, List<DiscretePredictionStatistics>>()
for (String method : methods)
	results.put(method, new ArrayList<DiscretePredictionStatistics>())

for (int fold = 0; fold < folds; fold++) {
	/*
	 * ADD EXTERNALLY COMPUTED EVIDENCE
	 */
	// do Naive Bayes training
	NaiveBayesUtil nb = new NaiveBayesUtil()

	log.debug("Fold {} Naive Bayes training set has {} documents", fold, nbTrainingKeys.get(fold).size())
	nb.learn(nbTrainingKeys.get(fold), dataPath + labelFile, dataPath + wordFile)

	inserter = data.getInserter(ClassifyCat, trainReadPartitions.get(fold))
	//	nb.insertAllPredictions(dataPath + wordFile, trainingKeys.get(fold), inserter)
	nb.insertAllProbabilities(dataPath + wordFile, trainingKeys.get(fold), inserter)
	log.debug("training keys size {}", trainingKeys.get(fold).size())

	inserter = data.getInserter(ClassifyCat, testReadPartitions.get(fold))
	//	nb.insertAllPredictions(dataPath + wordFile, testingKeys.get(fold), inserter)
	nb.insertAllProbabilities(dataPath + wordFile, testingKeys.get(fold), inserter)
	log.debug("testing keys size {}", testingKeys.get(fold).size())

	inserter = data.getInserter(Cat, trainReadPartitions.get(fold))
	for (int i=0; i < numCategories; i++) inserter.insert(data.getUniqueID(i));
	inserter = data.getInserter(Cat, testReadPartitions.get(fold))
	for (int i=0; i < numCategories; i++) inserter.insert(data.getUniqueID(i));


	/*
	 * POPULATE DATABASE
	 */

	def targetPredicates = [HasCat] as Set

	Map<Variable, Set<GroundTerm>> substitutions = new HashMap<Variable, Set<GroundTerm>>()
	Variable Category = new Variable("Category")
	Set<GroundTerm> categoryGroundings = new HashSet<GroundTerm>()
	for (int i = 0; i <= numCategories; i++)
		categoryGroundings.add(data.getUniqueID(i))
	substitutions.put(Category, categoryGroundings)

	Variable Document = new Variable("Document")
	Database db = data.getDatabase(trainReadPartitions.get(fold));
	DatabasePopulator dbPop = new DatabasePopulator(db);
	substitutions.put(Document, partitionDocuments.get(trainReadPartitions.get(fold)))
	dbPop.populate(new QueryAtom(avgValue__1, Document, Category), substitutions)
	db.close();
	
	db = data.getDatabase(testReadPartitions.get(fold));
	dbPop = new DatabasePopulator(db);
	substitutions.put(Document, partitionDocuments.get(testReadPartitions.get(fold)))
	dbPop.populate(new QueryAtom(avgValue__1, Document, Category), substitutions)
	db.close();
	
	/* open databases */

	toClose = [Link, ClassifyCat, Cat] as Set;
	Database trainDB = data.getDatabase(trainWritePartitions.get(fold), toClose, trainReadPartitions.get(fold))
	Database testDB = data.getDatabase(testWritePartitions.get(fold), toClose, testReadPartitions.get(fold))


	dbPop = new DatabasePopulator(trainDB)
	substitutions.put(Document, partitionDocuments.get(trainReadPartitions.get(fold)))
	dbPop.populate(new QueryAtom(HasCat, Document, Category), substitutions)

	/**
	 * POPULATE TEST DATABASE
	 */

	dbPop = new DatabasePopulator(testDB)
	substitutions.put(Document, partitionDocuments.get(testReadPartitions.get(fold)))
	dbPop.populate(new QueryAtom(HasCat, Document, Category), substitutions)


	Database labelsDB = data.getDatabase(trainLabelPartitions.get(fold), targetPredicates)

	// get all default weights
	Map<CompatibilityKernel,Weight> weights = new HashMap<CompatibilityKernel, Weight>()
	for (CompatibilityKernel k : Iterables.filter(m.getKernels(), CompatibilityKernel.class))
		weights.put(k, k.getWeight());

	def groundTruthDB = data.getDatabase(testLabelPartitions.get(fold), targetPredicates)
	DataOutputter.outputPredicate("output/graph/groundTruth" + fold + ".node" , groundTruthDB, HasCat, ",", false, "nodeid,label")
	groundTruthDB.close()

	DataOutputter.outputPredicate("output/graph/groundTruth" + fold + ".directed" , testDB, Link, ",", false, null)

	for (String method : methods) {
		for (CompatibilityKernel k : Iterables.filter(m.getKernels(), CompatibilityKernel.class))
			k.setWeight(weights.get(k))

		/*
		 * Weight learning
		 */
		learn(m, trainDB, labelsDB, wikiBundle, method, log)

		System.out.println("Learned model " + method + "\n" + m.toString())

		/*
		 * Inference on test set
		 */
		Set<GroundAtom> allAtoms = Queries.getAllAtoms(testDB, HasCat)
		for (RandomVariableAtom atom : Iterables.filter(allAtoms, RandomVariableAtom))
			atom.setValue(0.0)
		MPEInference mpe = new MPEInference(m, testDB, wikiBundle)
		FullInferenceResult result = mpe.mpeInference()
		System.out.println("Objective: " + result.getTotalWeightedIncompatibility())

		/*
		 * Evaluation
		 */
		def comparator = new DiscretePredictionComparator(testDB)
		groundTruthDB = data.getDatabase(testLabelPartitions.get(fold), [HasCat] as Set)
		comparator.setBaseline(groundTruthDB)
		comparator.setResultFilter(new MaxValueFilter(HasCat, 1))
		comparator.setThreshold(Double.MIN_VALUE) // treat best value as true as long as it is nonzero

		int totalTestExamples = testingKeys.get(fold).size() * numCategories;
		//System.out.println("totalTestExamples " + totalTestExamples)
		DiscretePredictionStatistics stats = comparator.compare(HasCat, totalTestExamples)
		System.out.println("F1 score " + stats.getF1(
				DiscretePredictionStatistics.BinaryClass.POSITIVE))

		results.get(method).add(fold, stats)

		DataOutputter.outputClassificationPredictions("output/" + method + fold + ".csv", testDB, HasCat, ",")

		groundTruthDB.close()
	}
	trainDB.close()
}

for (String method : methods) {
	def methodStats = results.get(method)
	for (int fold = 0; fold < folds; fold++) {
		def stats = methodStats.get(fold)
		def b = DiscretePredictionStatistics.BinaryClass.POSITIVE
		System.out.println("Method " + method + ", fold " + fold +", acc " + stats.getAccuracy() +
				", prec " + stats.getPrecision(b) + ", rec " + stats.getRecall(b) +
				", F1 " + stats.getF1(b) + ", correct " + stats.getCorrectAtoms().size() +
				", tp " + stats.tp + ", fp " + stats.fp + ", tn " + stats.tn + ", fn " + stats.fn)
	}
}


private void learn(Model m, Database db, Database labelsDB, ConfigBundle config, String method, Logger log) {
	switch(method) {
		case "MLE":
			MaxLikelihoodMPE mle = new MaxLikelihoodMPE(m, db, labelsDB, config)
			mle.learn()
			break
		case "MPLE":
			MaxPseudoLikelihood mple = new MaxPseudoLikelihood(m, db, labelsDB, config)
			mple.learn()
			break
		case "MM0.1":
			config.setProperty(MaxMargin.SLACK_PENALTY_KEY, 0.1);
			MaxMargin mm = new MaxMargin(m, db, labelsDB, config)
			mm.learn()
			break
		case "MM1":
			config.setProperty(MaxMargin.SLACK_PENALTY_KEY, 1);
			MaxMargin mm = new MaxMargin(m, db, labelsDB, config)
			mm.learn()
			break
		case "MM10":
			config.setProperty(MaxMargin.SLACK_PENALTY_KEY, 10);
			MaxMargin mm = new MaxMargin(m, db, labelsDB, config)
			mm.learn()
			break
		case "MM100":
			config.setProperty(MaxMargin.SLACK_PENALTY_KEY, 100);
			MaxMargin mm = new MaxMargin(m, db, labelsDB, config)
			mm.learn()
			break
		case "MM1000":
			config.setProperty(MaxMargin.SLACK_PENALTY_KEY, 1000);
			MaxMargin mm = new MaxMargin(m, db, labelsDB, config)
			mm.learn()
			break
		case "HEMRandOM":
			HardEMRandOM hardRandOM = new HardEMRandOM(m, db, labelsDB, config)
			hardRandOM.setSlackPenalty(10000)
		//hardRandOM.learn()
			break
		case "RandOM":
			FirstOrderMetropolisRandOM randOM = new FirstOrderMetropolisRandOM(m, db, labelsDB, config)
			randOM.learn()
			break
		case "SET_TO_ONE":
			for (CompatibilityKernel k : Iterables.filter(m.getKernels(), CompatibilityKernel.class))
				k.setWeight(new PositiveWeight(1.0))
		case "NONE":
			break;
		default:
			log.error("Invalid method ")
	}
}