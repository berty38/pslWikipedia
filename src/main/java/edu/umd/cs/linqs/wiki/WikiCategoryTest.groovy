package edu.umd.cs.linqs.wiki

import java.util.Set;

import org.slf4j.Logger
import org.slf4j.LoggerFactory

import edu.emory.mathcs.utils.ConcurrencyUtils
import edu.umd.cs.psl.application.inference.MPEInference
import edu.umd.cs.psl.application.learning.weight.VotedPerceptron
import edu.umd.cs.psl.config.*
import edu.umd.cs.psl.core.*
import edu.umd.cs.psl.core.inference.*
import edu.umd.cs.psl.database.DataStore
import edu.umd.cs.psl.database.Database
import edu.umd.cs.psl.database.DatabasePopulator
import edu.umd.cs.psl.database.DatabaseQuery
import edu.umd.cs.psl.database.Partition
import edu.umd.cs.psl.database.rdbms.RDBMSDataStore
import edu.umd.cs.psl.database.rdbms.driver.H2DatabaseDriver
import edu.umd.cs.psl.database.rdbms.driver.H2DatabaseDriver.Type
import edu.umd.cs.psl.evaluation.result.*
import edu.umd.cs.psl.evaluation.statistics.DiscretePredictionComparator
import edu.umd.cs.psl.evaluation.statistics.DiscretePredictionStatistics
import edu.umd.cs.psl.groovy.*
import edu.umd.cs.psl.model.Model;
import edu.umd.cs.psl.model.argument.ArgumentType
import edu.umd.cs.psl.model.argument.GroundTerm
import edu.umd.cs.psl.model.argument.Variable
import edu.umd.cs.psl.model.atom.QueryAtom
import edu.umd.cs.psl.model.function.AttributeSimilarityFunction
import edu.umd.cs.psl.ui.loading.*

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
m.add rule : ( HasCat(B,C) & Link(A,B) & (A - B) ) >> HasCat(A,C), weight: 1.0
m.add rule : ( HasCat(B,C) & Link(B,A) & (A - B) ) >> HasCat(A,C), weight: 1.0
m.add rule : ( Talk(D,A) & Talk(E,A) & HasCat(E,C) & (E - D) ) >> HasCat(D,C), weight: 1.0

m.add PredicateConstraint.PartialFunctional , on : HasCat

Partition fullObserved =  new Partition(0)
Partition groundTruth = new Partition(1)

def dataPath = "./data/"
def inserter
inserter = data.getInserter(Link, fullObserved)
InserterUtils.loadDelimitedData(inserter, dataPath + "uniqueLinks.txt")
inserter = data.getInserter(Talk, fullObserved)
InserterUtils.loadDelimitedData(inserter, dataPath + "talk.txt")

folds = 2
trainReadPartitions = new ArrayList<Partition>()
testReadPartitions = new ArrayList<Partition>()
trainWritePartitions = new ArrayList<Partition>()
testWritePartitions = new ArrayList<Partition>()
trainLabelPartitions = new ArrayList<Partition>()

def keys = new HashSet<Variable>()
Variable document = new Variable("Document")
Variable linkedDocument = new Variable("LinkedDoc")
def queries = new HashSet<DatabaseQuery>()
queries.add(new DatabaseQuery(ClassifyCat(document,N).getFormula()))
queries.add(new DatabaseQuery(Link(document, linkedDocument).getFormula()))
queries.add(new DatabaseQuery(Talk(document, A).getFormula()))
keys.add(document)
keys.add(linkedDocument)
def partitionDocuments = new HashMap<Partition, Set<GroundTerm>>()

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
}

Random rand = new Random()

double nbTrainingRatio = 0.3
// do Naive Bayes training
for (int fold = 0; fold < folds; fold++) {
	Partition observedPartition = trainReadPartitions.get(fold)

	NaiveBayesUtil nb = new NaiveBayesUtil()

	Set<Integer> trainingKeys = new HashSet<Integer>()
	Set<Integer> foldKeys = new HashSet<Integer>()

	for (GroundTerm doc : partitionDocuments.get(observedPartition)) {
		if (rand.nextDouble() < nbTrainingRatio)
			trainingKeys.add(Integer.decode(doc.toString()))
		foldKeys.add(Integer.decode(doc.toString()))
	}

	log.debug("training set unique IDS: " + trainingKeys)
	nb.learn(trainingKeys, "data/newCategoryBelonging.txt", "data/pruned-document.txt")

	inserter = data.getInserter(ClassifyCat, observedPartition)
	nb.insertAllProbabilities( "data/pruned-document.txt", foldKeys, inserter)

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
	VotedPerceptron vp = new VotedPerceptron(m, db, labelsDB, wikiBundle)
	vp.learn()
	
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
	def groundTruthDB = data.getDatabase(groundTruth)
	comparator.setBaseline(groundTruthDB)
	
	DiscretePredictionStatistics stats = comparator.compare(HasCat)
	System.out.println("F1 score " + stats.getF1(
		DiscretePredictionStatistics.BinaryClass.POSITIVE))
	
	db.close()
	groundTruthDB.close()
}