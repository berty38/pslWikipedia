package edu.umd.cs.psl.experiments.wikipedia;


import java.sql.Connection;
import java.util.HashSet;
import java.util.Set;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import de.mathnbits.experiment.Measure;

import edu.umd.cs.psl.config.PSLCoreConfiguration;
import edu.umd.cs.psl.config.WeightLearningConfiguration;
import edu.umd.cs.psl.core.ModelApplication;
import edu.umd.cs.psl.experiments.AbstractRDBMSExperiment;
import edu.umd.cs.psl.learning.weight.WeightLearningGlobalOpt;
import edu.umd.cs.psl.model.Model;
import edu.umd.cs.psl.model.argument.type.ArgumentType;
import edu.umd.cs.psl.model.kernel.Kernel;
import edu.umd.cs.psl.model.kernel.softrule.SoftRuleKernel;
import edu.umd.cs.psl.model.predicate.StandardPredicate;
import edu.umd.cs.psl.model.predicate.Predicate;
import edu.umd.cs.psl.model.predicate.PredicateFactory;
import edu.umd.cs.psl.ui.experiment.report.ExperimentReport;
import edu.umd.cs.psl.ui.functions.textsimilarity.WordClassifier;


abstract class AbstractWikipediaExperiment extends AbstractRDBMSExperiment {

	protected static final Logger log = LoggerFactory.getLogger(WikipediaCategoryExperiment.class);
	
	protected SoftRuleKernel[] rules;
	protected Set<Predicate> predicatesToClose;
	protected StandardPredicate document, similarText, category, intLink, edit, talk, usertalk, hasCat, unknown, known;
	
	
	public AbstractWikipediaExperiment(Connection db) {
		super(db);
		predicatesToClose = new HashSet<Predicate>();
	}
	

	@Override
	public void loadModel(Model model, PredicateFactory predFac) {
		unknown = predFac.createStandardPredicate("unknown",	new ArgumentType[]{ArgumentType.Entity});
		known = predFac.createStandardPredicate("known",	new ArgumentType[]{ArgumentType.Entity});
		document = predFac.createStandardPredicate("document",	new ArgumentType[]{ArgumentType.Entity, ArgumentType.Attribute, ArgumentType.Attribute});
		similarText = predFac.createStandardPredicate("similartext", new ArgumentType[]{ArgumentType.Entity, ArgumentType.Entity});
		category = predFac.createStandardPredicate("category", new ArgumentType[]{ArgumentType.Entity, ArgumentType.Attribute});
		intLink = predFac.createStandardPredicate("withinwithinlink",	new ArgumentType[]{ArgumentType.Entity, ArgumentType.Entity});
		edit = predFac.createStandardPredicate("editevent", new ArgumentType[]{ArgumentType.Entity, ArgumentType.Entity});
		talk = predFac.createStandardPredicate("talkevent", new ArgumentType[]{ArgumentType.Entity, ArgumentType.Entity});
		usertalk = predFac.createStandardPredicate("usertalk",	new ArgumentType[]{ArgumentType.Entity, ArgumentType.Entity});
		hasCat = predFac.createStandardPredicate("categorybelonging", new ArgumentType[]{ArgumentType.Entity, ArgumentType.Entity});
	}
	
	public void runInference(WikiLoadConfig config, int numfolds, ExperimentReport report, PSLCoreConfiguration pslconfig) {	
		report.writeObject("wikiconfig", config);
		Measure[] precisionRecall = new Measure[numfolds];
		
		for (int fold=0;fold<numfolds;fold++) {
			log.debug("Starting to load data for fold {} ...",fold);
			WikipediaSingleFoldLoader loader = new WikipediaSingleFoldLoader(config);
			setup();
			if (fold==0) report.writeModel(getModel());
			loader.loadTest(getDataLoader());
			ModelApplication app = getApplication(10, loader.getTestDataIDs(),pslconfig);
			app.totalMap();
			precisionRecall[fold]=precisionRecall(10,4,5);
			report.writeResultLn(precisionRecall[fold]);
		}
		report.writeResultLn(Measure.summaryStats(precisionRecall));
	}
	
	public void learn(WikiLoadConfig config, int numfolds, ExperimentReport report, WeightLearningConfiguration pslconfig) {
		report.writeObject("wikiconfig", config);
		Measure[] precisionRecall = new Measure[numfolds];
			
		for (int fold=0;fold<numfolds;fold++) {
			log.debug("Starting to learn for fold {} ...",fold);
			WikipediaSingleFoldLoader loader = new WikipediaSingleFoldLoader(config);
			setup();
			
			loader.loadTrain(getDataLoader());
			WeightLearningGlobalOpt w = new WeightLearningGlobalOpt(getModel(),
													this.getDatabase(101, loader.getTrainDataGroundTruthIDs(), predicatesToClose),
													this.getDatabase(102, loader.getTrainDataIDs(), new HashSet<Predicate>(0)),
													pslconfig);
			w.learn();
			report.writeModel(getModel(),"fold"+fold);
			setupDatabase();
			
			loader.loadTest(getDataLoader());
			ModelApplication app = getApplication(10, loader.getTestDataIDs(),pslconfig);
			app.totalMap();
			precisionRecall[fold]=precisionRecall(10,4,5);
			report.writeResultLn(precisionRecall[fold]);
		}
		report.writeResultLn(Measure.summaryStats(precisionRecall));
	}
	
	public void learnOnce(WikiLoadConfig config, int numfolds, ExperimentReport report, WeightLearningConfiguration pslconfig) {
		report.writeObject("wikiconfig", config);
		Measure[] precisionRecall = new Measure[numfolds];
		
		WikipediaSingleFoldLoader loaderTrain = new WikipediaSingleFoldLoader(config);
		setup();
		
		loaderTrain.loadTrain(getDataLoader());
		WeightLearningGlobalOpt w = new WeightLearningGlobalOpt(getModel(),
												this.getDatabase(101, loaderTrain.getTrainDataGroundTruthIDs(), predicatesToClose),
												this.getDatabase(102, loaderTrain.getTrainDataIDs(), new HashSet<Predicate>(0)),
												pslconfig);
		w.learn();
		report.writeModel(getModel());
		loaderTrain=null;
		
		for (int fold=0;fold<numfolds;fold++) {
			log.debug("Starting to apply to fold {} ...",fold);
			WikipediaSingleFoldLoader loader = new WikipediaSingleFoldLoader(config);
			setupDatabase();
			
			loader.loadTest(getDataLoader());
			ModelApplication app = getApplication(10, loader.getTestDataIDs(),pslconfig);
			app.totalMap();
			precisionRecall[fold]=precisionRecall(10,4,5);
			report.writeResultLn(precisionRecall[fold]);
		}
		report.writeResultLn(Measure.summaryStats(precisionRecall));

	}
	
	public abstract Measure precisionRecall(int writeID, int dataID, int truthID);
	


}
