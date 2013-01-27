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


public class WikipediaEditTest {

	WikipediaEditExperiment experiment;
	
	final WikiLoadConfig wikiconfig = new WikiLoadEditConfig();
	WeightLearningConfiguration config;
	ExperimentReport report;
	
	@Before
	public void setUp() throws Exception {
		Class.forName("org.h2.Driver");
		config = new WeightLearningConfiguration();
		//Connection conn = DatabaseUtil.getDiskDB("~/Development/Eclipse/PSL/sql/","wiki");
		Connection conn = DatabaseUtil.getMemoryDB("wiki");
		experiment = new WikipediaEditExperiment(conn);
		report = new ExperimentReport("~/Development/Eclipse/PSL/results/wikieditpredict/",true);
	}
	
	@Test
	public void loadTest() {
		WikipediaSingleFoldLoader loader = new WikipediaSingleFoldLoader(wikiconfig);
		experiment.setup();
		loader.loadTest(experiment.getDataLoader());
	}
	
	
	@Test
	public void inference() {
		config.setContinuousVariables(true);
		config.setNorm(DistanceNorm.L1);
		report.writeConfiguration(config, "coreconfiguration.config");
		experiment.runInference(wikiconfig,5,report,config);
	}

	@Test
	public void learn() {
		config.setContinuousVariables(true);
		config.setNorm(DistanceNorm.L1);
		config.setPointMoveConvergenceThres(-1); //Disable premature convergence
		config.setParameterPrior(5);
		config.setInitialParameter(1.0);
		config.setLearningType(Type.BFGS);
		report.writeConfiguration(config, "coreconfiguration.config");
		experiment.learnOnce(wikiconfig,5,report,config);
	}

	
	@After
	public void tearDown() throws Exception {
		experiment.close();
		report.close();
	}
	
}
