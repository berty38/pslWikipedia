package edu.umd.cs.psl.experiments.wikiclass;

import java.util.Set;

import edu.umd.cs.psl.groovy.*;
import edu.umd.cs.psl.ui.experiment.report.ExperimentReport;
import edu.umd.cs.psl.database.RDBMS.DatabaseDriver;
import edu.umd.cs.psl.model.function.AttributeSimilarityFunction;
import edu.umd.cs.psl.ui.functions.textsimilarity.*;
import edu.umd.cs.psl.learning.weight.WeightLearningGlobalOpt;
import edu.umd.cs.psl.config.*;

import edu.umd.cs.psl.ui.experiment.report.ExperimentReport;
import edu.umd.cs.psl.ui.experiment.report.StatisticsUtil;



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

//def basedir = "/Users/matthias/Development/Eclipse/PSL/data/wikipedia/nips2010data/";
//def outputdir = "/Users/matthias/Development/Eclipse/PSL/data/wikipedia/nips2010results";
//def datadir = "30foldsP=0.2C7";
//int totalFolds = 30;
//int[] foldRange = [1,1];
//int[] sampleSizes = [20000, 10000];
def basedir = args[0]
def outputdir = args[1]
def datadir = args[2];
int totalFolds = Integer.parseInt(args[3])
int[] foldRange = WikiClassUtil.parseInteger(args[4].split(","))
assert foldRange.length==2 && foldRange[0]<=foldRange[1] : Arrays.toString(foldRange);
int[] sampleSizes = WikiClassUtil.parseInteger(args[5].split(","))
assert sampleSizes.length>0;

noHistogramBins = 10;



def wikiconfig = new WikiLoadConfig(basedir);
WeightLearningConfiguration config = new WeightLearningConfiguration();
//config.setLearningType(WeightLearningGlobalOpt.Type.BFGS);
//config.setInitialParameter(1.0);
//config.setParameterPrior(4.0);
config.setPerceptronIterations(10);
config.setPerceptronUpdateFactor(0.05);

int trainID = 1;
int trainTruthID = 9;
int testID1 = 11;
int testID2 = 12;
int testTruthID = 19;
int neutralID = 0;



for (int fold=foldRange[0]; fold<=foldRange[1]; fold++) {
	DataStore data = new RelationalDataStore(m)
	data.setup db : DatabaseDriver.H2 , type: "memory"
	
	ExperimentReport report = new ExperimentReport(outputdir,fold+"of"+datadir,true);
	
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

	m.learn data, evidence : [neutralID,trainID], infered: trainTruthID, close : category, config : config
	
	//Write report
	report.writeModel(m.getModel());
	report.writeObject("wikiconfiguration.txt",wikiconfig);
	report.writeConfiguration(config,"coreconfiguration.txt");
	
	//Do inference
	
	def result
	result = m.mapInference(data.getDatabase(write: 100, parts : [neutralID,testID1]))
	report.writeResultLn("Baseline Accuracy (using only document text):");
	report.writeResultLn("@ " + WikiClassUtil.computeAccuracy(data,100,testTruthID));
	result = m.mapInference(data.getDatabase(write: 101, parts : [neutralID,testID1,testID2]))
	report.writeResultLn("Baseline Accuracy (using links & text):");
	report.writeResultLn("@ " + WikiClassUtil.computeAccuracy(data,101,testTruthID));
	
	report.writeResultLn("@ ---------------");
	
	int counter = 0;
	def baseResult = null;
	for (int sampleSize : sampleSizes) {
		int mwriteID = 200+counter;
		report.writeResultLn("Marginal Computing for sample size "+sampleSize);
		config.setSamplingSteps(sampleSize);
		
		result = m.marginalInference(data.getDatabase(write: mwriteID, parts : [neutralID,testID1,testID2]), config)
		report.writeObject("samplerConfiguration"+sampleSize,result.getSamplingStatistics());
		result.printHistograms(category)

		report.writeResultLn("Accuracy of Marginals with "+sampleSize+":");
		report.writeResultLn("@ " + WikiClassUtil.computeAccuracy(data,mwriteID,testTruthID));
		
		
		double[] posConfidences = data.queryStats("SELECT h.confidence as conf FROM category h, category k WHERE " +
							" h.fold="+mwriteID+" and h.truth=(select max(truth) from category as f where f.paper=h.paper and f.fold="+mwriteID+") " +
							"and k.fold="+testTruthID+" and k.paper=h.paper and k.category=h.category and h.truth>0.2",   "conf");
		double[] negConfidences = data.queryStats("SELECT h.confidence as conf FROM category h, category k WHERE " +
				" h.fold="+mwriteID+" and h.truth=(select max(truth) from category as f where f.paper=h.paper and f.fold="+mwriteID+") " +
				"and k.fold="+testTruthID+" and k.paper=h.paper and k.category!=h.category and h.truth>0.2",   "conf");
		
	
		report.writeResultLn("Average positive confidence:");
		report.writeResultLn("@ " + StatisticsUtil.average(posConfidences));
		report.writeResultLn(Arrays.toString(posConfidences));
		report.writeResultLn("Average negative confidence:");
		report.writeResultLn("@ " + StatisticsUtil.average(negConfidences));
		report.writeResultLn(Arrays.toString(negConfidences));
		
		if (counter==0) baseResult = result;
		else {
			assert baseResult!=null;
			report.writeResultLn("KL divergence to first sample:");
			report.writeResultLn("@ " + baseResult.averageKLdivergence(category,noHistogramBins,result));
		}
		
		report.writeResultLn("@ ---------------");
		counter++;
	}
	
	data.close();
	report.close();
}

//double[][] accuracy = new double[foldRange[1]-foldRange[0]+1][3];
//println "Accuracies"
//double[] runningAcc = new double[3];
//for (int i=0;i<noFolds;i++) {
//	println accuracy[i][0] + " vs. " + accuracy[i][1] + " vs. " + accuracy[i][2] + " => " + ((accuracy[i][1]/accuracy[i][0]-1)*100) + 
//														"   |   " + ((accuracy[i][2]/accuracy[i][1]-1)*100)+ " % improvement"
//	runningAcc[0]+=accuracy[i][0];
//	runningAcc[1]+=accuracy[i][1];
//	runningAcc[2]+=accuracy[i][2];
//}
//println "Avg accuracies: " + Arrays.toString(runningAcc);
//runningAcc[0]*=1.0/noFolds;
//runningAcc[1]*=1.0/noFolds;
//runningAcc[2]*=1.0/noFolds;
//
//println "Avg accuracies: " + Arrays.toString(runningAcc);
