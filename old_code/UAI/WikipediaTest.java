package edu.umd.cs.psl.experiments.wikipedia;

import static org.junit.Assert.assertEquals;

import java.sql.Connection;
import java.sql.DriverManager;

import org.junit.*;

import com.google.common.collect.ImmutableList;

import edu.umd.cs.psl.config.WeightLearningConfiguration;
import edu.umd.cs.psl.core.ModelApplication;
import edu.umd.cs.psl.database.RDBMS.RDBMSDatabase;
import edu.umd.cs.psl.experiments.DatabaseUtil;
import edu.umd.cs.psl.experiments.wikipedia.WikipediaCategoryExperiment.Mode;
import edu.umd.cs.psl.learning.weight.WeightLearningGlobalOpt.Type;
import edu.umd.cs.psl.model.DistanceNorm;
import edu.umd.cs.psl.model.argument.Term;
import edu.umd.cs.psl.model.argument.Variable;
import edu.umd.cs.psl.model.atom.Atom;
import edu.umd.cs.psl.model.atom.AtomStore;
import edu.umd.cs.psl.model.atom.VariableAssignment;
import edu.umd.cs.psl.model.formula.Conjunction;
import edu.umd.cs.psl.model.formula.Formula;
import edu.umd.cs.psl.model.predicate.PredicateFactory;
import edu.umd.cs.psl.ui.experiment.report.ExperimentReport;


public class WikipediaTest {

	WikipediaCategoryExperiment example;
	
	final WikiLoadCategoryConfig wikiconfig = new WikiLoadCategoryConfig();
	WeightLearningConfiguration config;
	ExperimentReport report;
	
	
	@Before
	public void setUp() throws Exception {
		setUp(Mode.ALT,false);
	}
	
	public void setUp(Mode m, boolean trainClassifier) throws Exception {
		Class.forName("org.h2.Driver");
		config = new WeightLearningConfiguration();
		//Connection conn = DatabaseUtil.getDiskDB("~/Development/Eclipse/PSL/sql/","wiki");
		Connection conn = DatabaseUtil.getMemoryDB("wiki");
		example = new WikipediaCategoryExperiment(conn,trainClassifier,m);
		report = new ExperimentReport("./results/wikicategory",true);
	}
	
	@Test
	public void computeSimilarity() {
		WikipediaSingleFoldLoader.computeTextSimilarity(wikiconfig, "document.txt", "documentSimilarity.txt", 0.1,7);
	}
	
	@Test
	public void loadTest() {
		WikipediaSingleFoldLoader loader = new WikipediaSingleFoldLoader(wikiconfig);
		example.setup();
		loader.loadTest(example.getDataLoader());
	}
	
	//@Test
	public void testSetup() {
		//example.runInference(wikiconfig,5,report);
		
		ModelApplication app = example.getApplication(1, new int[]{0});
		PredicateFactory predFac = example.getModel().getPredicateFactory();
		
		Atom query = new Atom(predFac.getPredicate("knows"),new Term[]{new Variable("A"),new Variable("B")});
		assertEquals(app.getDatabase().query(query).size(),8);
		Atom query2 = new Atom(predFac.getPredicate("name"),new Term[]{new Variable("A"),new Variable("B")});
		assertEquals(app.getDatabase().query(query2).size(),6);
		Atom queryP2 = new Atom(predFac.getPredicate("knows"),new Term[]{new Variable("B"),new Variable("C")});
		Formula query3 = new Conjunction(query,queryP2);
		assertEquals(app.getDatabase().query(query3).size(),8);
		
//		assertEquals(db.query(query2, store).size(),1);
//		assertEquals(db.query(query2, store, ImmutableList.of(new Variable("A"),new Variable("C") )).size(),1);

//		VariableAssignment assign = new VariableAssignment();
//		assign.assign(new Variable("B"), getEntity(2));
//		assertEquals(db.query(query, store,assign).size(),1);
//		assertEquals(db.query(query2, store, assign, ImmutableList.of(new Variable("A"),new Variable("C") )).size(),1);
		
		Atom query4P1 = new Atom(predFac.getPredicate("name"),new Term[]{new Variable("A"),new Variable("X")});
		Atom query4P2 = new Atom(predFac.getPredicate("name"),new Term[]{new Variable("B"),new Variable("Y")});
		Atom query4P3 = new Atom(predFac.getPredicate("sameName"),new Term[]{new Variable("X"),new Variable("Y")});
		Formula query4 = new Conjunction(query4P1,new Conjunction(query4P2,query4P3));
		//System.out.println(app.query(query4));
		assertEquals(app.getDatabase().query(query4).size(),10);
	}
	
	@Test
	public void inference() {
		config.setContinuousVariables(true);
		config.setNorm(DistanceNorm.L1);
		wikiconfig.percentageDocumentHoldout=1.0;
		
		report.writeConfiguration(config, "coreconfiguration.config");
		example.runInference(wikiconfig,1,report,config);
		
//		example.learn(path2Data);
//		ModelApplication app = example.getApplication(1, new int[]{0});
//		app.totalMap();
	}

	@Test
	public void learnOnce() {
		config.setContinuousVariables(true);
		config.setNorm(DistanceNorm.L1);
		config.setPointMoveConvergenceThres(-1); //Disable premature convergence
		config.setParameterPrior(5);
		config.setInitialParameter(1.0);
		config.setLearningType(Type.BFGS);
		report.writeConfiguration(config, "coreconfiguration.config");
		
		example.learnOnce(wikiconfig,5,report,config);
	}
	
	@Test
	public void learnEach() {
		config.setContinuousVariables(true);
		config.setNorm(DistanceNorm.L1);
		config.setPointMoveConvergenceThres(-1); //Disable premature convergence
		config.setParameterPrior(5);
		config.setInitialParameter(1.0);
		config.setLearningType(Type.BFGS);
		report.writeConfiguration(config, "coreconfiguration.config");
		example.learn(wikiconfig,10,report,config);
	}
	
	public void learnRepeated(double holdout, int runs) {
		config.setContinuousVariables(true);
		config.setNorm(DistanceNorm.L1);
		config.setPointMoveConvergenceThres(-1); //Disable premature convergence
		config.setParameterPrior(5);
		config.setInitialParameter(1.0);
		config.setLearningType(Type.BFGS);
		report.writeConfiguration(config, "coreconfiguration.config");
		
		wikiconfig.percentageDocumentHoldout=holdout;
		for (int i=0;i<4;i++)
			example.learnOnce(wikiconfig,4,report,config);
	}
	
	public void learnRepeatedClassify(double classify, int runs) {
		config.setContinuousVariables(true);
		config.setNorm(DistanceNorm.L1);
		config.setPointMoveConvergenceThres(-1); //Disable premature convergence
		config.setParameterPrior(5);
		config.setInitialParameter(1.0);
		config.setLearningType(Type.BFGS);
		report.writeConfiguration(config, "coreconfiguration.config");
		
		wikiconfig.percentageClassifyTrain=classify;
		wikiconfig.percentageDocumentHoldout=1.0;
		for (int i=0;i<runs;i++)
			example.learnOnce(wikiconfig,4,report,config);
	}

	@After
	public void tearDown() throws Exception {
		example.close();
		report.close();
	}
	
	public static void main(String[] args) throws Exception {
		double percentClassify = Double.parseDouble(args[1]);
		int runs = Integer.parseInt(args[2]);
		WikipediaTest test = new WikipediaTest();
		test.setUp(getMode(args[0]),true);
		test.learnRepeatedClassify(percentClassify,runs);
		//test.inference();
		test.tearDown();
	}
	
	public static Mode getMode(String m) {
		if (m.equalsIgnoreCase("A")) return Mode.A;
		else if (m.equalsIgnoreCase("AL")) return Mode.AL;
		else if (m.equalsIgnoreCase("AT")) return Mode.AT;
		else if (m.equalsIgnoreCase("ALT")) return Mode.ALT;
		else throw new IllegalArgumentException(m);
	}
	
}
