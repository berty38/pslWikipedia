package edu.umd.cs.psl.experiments.wikiclass;

import java.util.Set;

import edu.umd.cs.psl.groovy.*;
import edu.umd.cs.psl.ui.experiment.report.ExperimentReport;
import edu.umd.cs.psl.database.RDBMS.DatabaseDriver;
import edu.umd.cs.psl.model.function.AttributeSimilarityFunction;
import edu.umd.cs.psl.ui.functions.textsimilarity.*;
import edu.umd.cs.psl.learning.weight.WeightLearningGlobalOpt;
import edu.umd.cs.psl.config.*;



PSLModel m = new PSLModel(this);

m.add predicate: "category" , paper: Entity, category : Entity, open: true
m.add predicate: "ground" , paper: Entity, category : Entity
m.add predicate: "categoryname", category : Entity, name: Attribute
m.add predicate: "similarContent" , paper1: Entity, paper2 : Entity
m.add predicate: "known", paper: Entity
m.add predicate: "unknown", paper: Entity

m.add predicate: "links", citing: Entity, cited: Entity

m.add rule : ( ground(A,C) ) >> category(A,C), weight: 0

m.add rule : ( category(A,C) & similarContent(A,B) & unknown(B)) >> category(B,C), weight : 5
m.add rule : ( category(A,C) & similarContent(B,A) & unknown(B)) >> category(B,C), weight : 5.5

//m.add rule : ( category(A,C) & links(B,A) & categoryname(C,'Genetic_Algorithms') ) >> category(B,C),  weight : 20
//m.add rule : ( category(A,C) & links(A,B)  & categoryname(C,'Genetic_Algorithms') ) >> category(B,C),  weight : 30

m.add rule : ( category(A,C) & links(B,A) & unknown(B) ) >> category(B,C),  weight : 13
m.add rule : ( category(A,C) & links(A,B) & unknown(B) ) >> category(B,C),  weight : 13


m.add PredicateConstraint.Functional , on : category


//Let's see what our model looks like.
println m;

def basedir = "/Users/matthias/Development/Eclipse/PSL/data/wikipedia/nips2010data/";
def outputdir = "/Users/matthias/Development/Eclipse/PSL/data/wikipedia/nips2010data/results";
def datadir = "20foldsP=0.2C7";
int totalFolds = 20;
int[] foldRange = [1,20];
int[] sampleSizes = [10000, 20000];



def wikiconfig = new WikiLoadConfig(basedir);
WeightLearningConfiguration config = new WeightLearningConfiguration();
//config.setLearningType(WeightLearningGlobalOpt.Type.BFGS);
//config.setInitialParameter(1.0);
//config.setParameterPrior(4.0);

int trainID = 1;
int trainTruthID = 9;
int testID1 = 11;
int testID2 = 12;
int testTruthID = 19;
int neutralID = 0;

int noFolds = foldRange[1]-foldRange[0]+1;
double[][] accuracy = new double[noFolds][3];

for (int fold=foldRange[0]; fold<=foldRange[1]; fold++) {
	DataStore data = new RelationalDataStore(m)
	data.setup db : DatabaseDriver.H2 , type: "memory"
	
//	ExperimentReport report = new ExperimentReport(outputdir,fold+"of"+datadir);
	
	def knownIns, unknownIns, catIns, linkIns, textIns, groundIns;
	def catNameIns;
	Set<Integer> knownIDs, unknownIDs, ids;
	
	//Read training data
	knownIns = data.getInserter(known,trainID);
	unknownIns = data.getInserter(unknown,trainID);
	knownIDs = WikiClassUtil.readDocumentFile(wikiconfig.getTrainPath(datadir,wikiconfig.knownFile,fold,totalFolds),knownIns);
	unknownIDs = WikiClassUtil.readDocumentFile(wikiconfig.getTrainPath(datadir,wikiconfig.unknownFile,fold,totalFolds),unknownIns);
	//groundIns = data.getInserter(ground,trainID);
	//WikiClassUtil.fullGrounding(wikiconfig.validCategories,unknownIDs,groundIns);
	
	catIns = data.getInserter(category,trainID);
	WikiClassUtil.readCategoryFile(wikiconfig.getPath(wikiconfig.categoryBelongFile),catIns,knownIDs);
	catIns = data.getInserter(category,trainTruthID);
	WikiClassUtil.readCategoryFile(wikiconfig.getPath(wikiconfig.categoryBelongFile),catIns,unknownIDs);
	
	ids = new HashSet<Integer>(); ids.addAll(knownIDs); ids.addAll(unknownIDs);
	linkIns = data.getInserter(links,trainID);
	WikiClassUtil.readRelationFile wikiconfig.getPath(wikiconfig.linksFile), linkIns, ids, false;
	textIns = data.getInserter(similarContent,trainID);
	WikiClassUtil.readRelationFile wikiconfig.getPath(wikiconfig.textFile), textIns, ids, true;

	
	//Read test data
	knownIns = data.getInserter(known,testID1);
	unknownIns = data.getInserter(unknown,testID1);
	knownIDs = WikiClassUtil.readDocumentFile(wikiconfig.getTestPath(datadir,wikiconfig.knownFile,fold,totalFolds),knownIns);
	unknownIDs = WikiClassUtil.readDocumentFile(wikiconfig.getTestPath(datadir,wikiconfig.unknownFile,fold,totalFolds),unknownIns);
	groundIns = data.getInserter(ground,testID1);
	WikiClassUtil.fullGrounding(wikiconfig.validCategories,unknownIDs,groundIns);
	
	
	catIns = data.getInserter(category,testID1);
	WikiClassUtil.readCategoryFile(wikiconfig.getPath(wikiconfig.categoryBelongFile),catIns,knownIDs);
	catIns = data.getInserter(category,testTruthID);
	WikiClassUtil.readCategoryFile(wikiconfig.getPath(wikiconfig.categoryBelongFile),catIns,unknownIDs);
	
	ids = new HashSet<Integer>(); ids.addAll(knownIDs); ids.addAll(unknownIDs);
	linkIns = data.getInserter(links,testID2);
	WikiClassUtil.readRelationFile wikiconfig.getPath(wikiconfig.linksFile), linkIns, ids, false;
	textIns = data.getInserter(similarContent,testID1);
	WikiClassUtil.readRelationFile wikiconfig.getPath(wikiconfig.textFile), textIns, ids, true;
	
	
	//Insert categories
	catIns = data.getInserter(categoryname,neutralID)
	WikiClassUtil.insertCategories(wikiconfig.validCategories, catIns);
	
	//Learn

//	m.learn data, evidence : [neutralID,trainID], infered: trainTruthID, close : category, config : config
	
	//Write report
//	report.writeModel(m);
//	report.writeObject("wikiconfiguration.txt",wikiconfig);
//	report.writeConfiguration(config,"coreconfiguration.txt");
	
	//Do inference
	
	def result
	result = m.mapInference(data.getDatabase(write: 100, parts : [neutralID,testID1]))
	result = m.mapInference(data.getDatabase(write: 101, parts : [neutralID,testID1,testID2]))
	

//	config.setSamplingSteps(10000);
//	result = m.marginalInference(data.getDatabase(write: 102, parts : [neutralID,testID1,testID2]), config)
//	println result.getSamplingStatistics().toString();
//	result.printHistograms(category)

	for (int writeID : [100,101,102]) {
		String var = "res";
		double numCorrect = data.querySingleStats("SELECT count(h.paper) as res FROM category h, category k WHERE " +
				" h.fold="+writeID+" and h.truth=(select max(truth) from category as f where f.paper=h.paper and f.fold="+writeID+") " +
				"and k.fold="+testTruthID+" and k.paper=h.paper and k.category=h.category and h.truth>0.2",
				var);
		double doccount = data.querySingleStats("SELECT COUNT(DISTINCT c.paper) as res from category c where c.fold="+testTruthID,var);
		//double[] numPredicted = queryStats("SELECT count(DISTINCT h.doc) as res FROM categorybelonging h WHERE " +
		//			" h.fold="+writeID, var);
		
		double acc = (numCorrect*1.0/doccount);
		accuracy[fold-1][writeID%3]=acc;
		System.out.println(writeID + ": | Num correct "+numCorrect+" of "+doccount+ " --> Accuracy: " + acc);
	}
	
	double[] posConfidences = data.queryStats("SELECT h.confidence as conf FROM category h, category k WHERE " +
						" h.fold="+102+" and h.truth=(select max(truth) from category as f where f.paper=h.paper and f.fold="+102+") " +
						"and k.fold="+testTruthID+" and k.paper=h.paper and k.category=h.category and h.truth>0.2",   "conf");
	double[] negConfidences = data.queryStats("SELECT h.confidence as conf FROM category h, category k WHERE " +
			" h.fold="+102+" and h.truth=(select max(truth) from category as f where f.paper=h.paper and f.fold="+102+") " +
			"and k.fold="+testTruthID+" and k.paper=h.paper and k.category!=h.category and h.truth>0.2",   "conf");
	

	
	println Arrays.toString(posConfidences);
	println Arrays.toString(negConfidences);
	
	data.close();
//	report.close();
}

println "Accuracies"
double[] runningAcc = new double[3];
int i=0;
for (int fold=foldRange[0]; fold<=foldRange[1]; fold++) {
	println accuracy[i][0] + " vs. " + accuracy[i][1] + " vs. " + accuracy[i][2] + " => " + ((accuracy[i][1]/accuracy[i][0]-1)*100) + 
														"   |   " + ((accuracy[i][2]/accuracy[i][1]-1)*100)+ " % improvement"
	runningAcc[0]+=accuracy[i][0];
	runningAcc[1]+=accuracy[i][1];
	runningAcc[2]+=accuracy[i][2];
	i++;
}
println "Avg accuracies: " + Arrays.toString(runningAcc);
runningAcc[0]*=1.0/noFolds;
runningAcc[1]*=1.0/noFolds;
runningAcc[2]*=1.0/noFolds;

println "Avg accuracies: " + Arrays.toString(runningAcc);

//result = m.marginalInference(data.getDatabase())

//result.printAtoms(samePerson,0.000001)

//result.printDistributions(samePerson)
//result.printHistograms(samePerson)