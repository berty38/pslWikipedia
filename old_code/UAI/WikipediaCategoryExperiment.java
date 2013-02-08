package edu.umd.cs.psl.experiments.wikipedia;

import java.sql.Connection;
import java.util.HashSet;
import java.util.Set;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import cern.colt.Arrays;

import com.google.common.collect.ImmutableSet;

import de.mathnbits.experiment.Measure;

import edu.umd.cs.psl.core.ModelApplication;
import edu.umd.cs.psl.database.RDBMS.RDBMSArgumentTypes;
import edu.umd.cs.psl.database.RDBMS.RDBMSPredicateHandle;
import edu.umd.cs.psl.experiments.AbstractRDBMSExperiment;
import edu.umd.cs.psl.learning.weight.WeightLearningGlobalOpt;
import edu.umd.cs.psl.model.DistanceNorm;
import edu.umd.cs.psl.model.Model;
import edu.umd.cs.psl.model.argument.Variable;
import edu.umd.cs.psl.model.argument.type.ArgumentType;
import edu.umd.cs.psl.model.formula.Conjunction;
import edu.umd.cs.psl.model.kernel.Kernel;
import edu.umd.cs.psl.model.kernel.predicateconstraint.PredicateConstraintType;
import edu.umd.cs.psl.model.kernel.predicateconstraint.PredicateConstraintKernel;
import edu.umd.cs.psl.model.kernel.priorweight.PriorWeightKernel;
import edu.umd.cs.psl.model.kernel.setdefinition.SetEntityDefinitionType;
import edu.umd.cs.psl.model.kernel.softrule.SoftRuleKernel;
import edu.umd.cs.psl.model.predicate.StandardPredicate;
import edu.umd.cs.psl.model.predicate.FunctionalPredicate;
import edu.umd.cs.psl.model.predicate.Predicate;
import edu.umd.cs.psl.model.predicate.PredicateFactory;
import edu.umd.cs.psl.model.predicate.SpecialPredicates;
import edu.umd.cs.psl.model.set.term.FormulaSetTerm;
import edu.umd.cs.psl.model.set.term.SetTerm;
import edu.umd.cs.psl.model.set.term.SetUnion;
import edu.umd.cs.psl.model.set.term.VariableSetTerm;
import edu.umd.cs.psl.ui.aggregators.AggregateSetAverage;
import edu.umd.cs.psl.ui.functions.textsimilarity.WordClassifier;
import edu.umd.cs.psl.ui.functions.textsimilarity.WordClassifier.ParseType;

public class WikipediaCategoryExperiment extends AbstractWikipediaExperiment {
	
	StandardPredicate nghCategory, sameLinks, similarUsers, similarEditors, userCategory, editCategory, talkCategory, nghText, classifyCat;
	FunctionalPredicate classify;
	
	public enum Mode { A , AL, AT, ALT };
	
	protected final boolean trainClassifier;
	private final Mode mode;

	
	public WikipediaCategoryExperiment(Connection db, boolean trainClassify, Mode m) {
		super(db);
		trainClassifier = trainClassify;
		mode = m;
		predicatesToClose = new HashSet<Predicate>();
	}
	
	@Override
	public void loadSchema() {
		registerPredicate(new RDBMSPredicateHandle(document, "document", new String[]{"doc", "text", "quality"}, true, "fold", null, null,null),
				RDBMSArgumentTypes.LongString);
		registerPredicate(new RDBMSPredicateHandle(similarText, "similartext", new String[]{"doc1", "doc2"}, true, "fold", "truth", null,null));
		registerPredicate(new RDBMSPredicateHandle(classifyCat, "classifyCat", new String[]{"doc", "category"}, true, "fold", "truth",null, null));
		registerPredicate(new RDBMSPredicateHandle(category, "category", new String[]{"category", "name"}, true, "fold", null, null,null));
		registerPredicate(new RDBMSPredicateHandle(intLink, "withinwithinlink", new String[]{"fromDoc", "toDoc"}, true, "fold", null, null,null));
		registerPredicate(new RDBMSPredicateHandle(edit, "editevent", new String[]{"doc", "user"}, true, "fold", "truth",null, null));
		registerPredicate(new RDBMSPredicateHandle(talk, "talkevent", new String[]{"doc", "user"}, true, "fold", "truth", null,null));
		registerPredicate(new RDBMSPredicateHandle(usertalk, "usertalk", new String[]{"user1", "user2"}, true, "fold", null,null, null));
		registerPredicate(new RDBMSPredicateHandle(unknown, "unknown", new String[]{"doc"}, true, "fold", null, null,null));
		registerPredicate(new RDBMSPredicateHandle(known, "known", new String[]{"doc"}, true, "fold", null,null, null));
		registerPredicate(new RDBMSPredicateHandle(hasCat, "categorybelonging", new String[]{"doc", "category"}, false, "fold", "truth",null, "psl"));
		registerPredicate(new RDBMSPredicateHandle(nghCategory, "nghcategory", new String[]{"doc", "category"}, false, "fold", "truth", null,"psl"));
		registerPredicate(new RDBMSPredicateHandle(sameLinks, "sameLinks", new String[]{"doc1", "doc2"}, false, "fold", "truth", null,"psl"));
		registerPredicate(new RDBMSPredicateHandle(similarUsers, "similarUsers", new String[]{"user1", "user2"}, false, "fold", "truth",null, "psl"));
		registerPredicate(new RDBMSPredicateHandle(similarEditors, "similarEditors", new String[]{"doc1", "doc2"}, false, "fold", "truth", null,"psl"));
		registerPredicate(new RDBMSPredicateHandle(userCategory, "userCategory", new String[]{"user", "category"}, false, "fold", "truth",null, "psl"));
		registerPredicate(new RDBMSPredicateHandle(editCategory, "editcategory", new String[]{"doc", "category"}, false, "fold", "truth",null, "psl"));
		registerPredicate(new RDBMSPredicateHandle(talkCategory, "talkcategory", new String[]{"doc", "category"}, false, "fold", "truth",null, "psl"));
		registerPredicate(new RDBMSPredicateHandle(nghText, "nghtext", new String[]{"doc", "category"}, false, "fold", "truth",null, "psl"));
	}

	@Override
	public void loadModel(Model model, PredicateFactory predFac) {
		super.loadModel(model, predFac);
		nghCategory = predFac.createStandardPredicate("nghcategory", new ArgumentType[]{ArgumentType.Entity, ArgumentType.Entity});
		sameLinks = predFac.createStandardPredicate("sameLinks", new ArgumentType[]{ArgumentType.Entity, ArgumentType.Entity});
		classifyCat = predFac.createStandardPredicate("classifyCat", new ArgumentType[]{ArgumentType.Entity, ArgumentType.Entity});
		similarUsers = predFac.createStandardPredicate("similarUsers",	new ArgumentType[]{ArgumentType.Entity, ArgumentType.Entity});
		similarEditors = predFac.createStandardPredicate("similarEditors",	new ArgumentType[]{ArgumentType.Entity, ArgumentType.Entity});
		userCategory = predFac.createStandardPredicate("userCategory",	new ArgumentType[]{ArgumentType.Entity, ArgumentType.Entity});
		editCategory = predFac.createStandardPredicate("editcategory",	new ArgumentType[]{ArgumentType.Entity, ArgumentType.Entity});
		talkCategory = predFac.createStandardPredicate("talkCategory",	new ArgumentType[]{ArgumentType.Entity, ArgumentType.Entity});
		nghText = predFac.createStandardPredicate("nghtext", new ArgumentType[]{ArgumentType.Entity, ArgumentType.Entity});
		
		//FunctionalPredicate sameText = predFac.createFunctionalPredicate("sameText", "cosine");
		
		//if (trainClassifier) classify = predFac.createFunctionalPredicate("classify", classifier);
		
		//Evidence
		predicatesToClose.add(hasCat);
		rules = new SoftRuleKernel[20];
		
		

		//============= Attribute Evidence =================
		//#Set version
//		SetTerm t0a = new FormulaSetTerm(new Conjunction(getVarAtom(similarText,"A","B"),new Conjunction(getVarAtom(SpecialPredicates.Unequal,"A","B"),getVarAtom(unknown,"A"))),new Variable("B"),ImmutableSet.of(new Variable("A")));
//		SetTerm t0b = new VariableSetTerm(new Variable("C"),ArgumentType.Entity);
//		SetEntityDefinitionType set0 = new SetEntityDefinitionType(nghText,t0a,t0b,new Variable[]{new Variable("A"), new Variable("C")},hasCat,"setaverage");
//		model.addModelEvidence(set0);
//		
//		rules[0] = new SoftRuleType(getVarAtom(nghText,"A","C"),
//									getVarAtom(hasCat,"A","C"),0.14);
//		model.addModelEvidence(rules[0]);
		
		//#Non-set version
//		rules[0] = new SoftRuleType(new Conjunction(new Conjunction(getVarAtom(hasCat,"B","C"),new Conjunction(getVarAtom(SpecialPredicates.Unequal,"A","B"),getVarAtom(unknown,"A"))),
//													getVarAtom(similarText,"A","B")),
//									getVarAtom(hasCat,"A","C"),1.0);
//		model.addModelEvidence(rules[0]);
		
		model.addKernel(new SoftRuleKernel(new Conjunction(getVarAtom(classifyCat,"A","C"),getVarAtom(unknown,"A")),
								getVarAtom(hasCat,"A","C"),1.0)
		);
		
				
		//============= Link Evidence =================
		//##Set Version
//		SetTerm t1a1 = new FormulaSetTerm(new Conjunction(getVarAtom(intLink,"A","X"),getVarAtom(unknown,"X")),new Variable("X"),ImmutableSet.of(new Variable("A")));
//		SetTerm t1a2 = new FormulaSetTerm(new Conjunction(getVarAtom(intLink,"X","A"),getVarAtom(unknown,"X")),new Variable("X"),ImmutableSet.of(new Variable("A")));
//		SetTerm t1a = new SetUnion(t1a1,t1a2);
//		SetTerm t1b = new VariableSetTerm(new Variable("C"),ArgumentType.Entity);
//		SetEntityDefinitionType set1 = new SetEntityDefinitionType(nghCategory,t1a,t1b,new Variable[]{new Variable("A"), new Variable("C")},hasCat,new AggregateSetAverage(1.0));
//		model.addModelEvidence(set1);

		
//		SetTerm t1a3 = new FormulaSetTerm(new Conjunction(new Conjunction(getVarAtom(talk,"A","U"),getVarAtom(unknown,"A")),
//				new Conjunction(getVarAtom(SpecialPredicates.Unequal,"A","X"),getVarAtom(talk,"X","U"))),
//				new Variable("X"),ImmutableSet.of(new Variable("A")));
//		SetTerm t1aa = new SetUnion(t1a,t1a3);
//		SetEntityDefinitionType set2 = new SetEntityDefinitionType(nghCategory,t1aa,t1b,new Variable[]{new Variable("A"), new Variable("C")},hasCat,"setaverage",false);
//		model.addModelEvidence(set2);
		
		
//		rules[2] = new SoftRuleType(getVarAtom(hasCat,"A","C"),
//									getVarAtom(nghCategory,"A","C"),1.0);
//		model.addModelEvidence(rules[2]);
		
		
		
		//#Non-set version
		if (mode==Mode.AL || mode==Mode.ALT) {
		rules[2] = new SoftRuleKernel(new Conjunction(new Conjunction(getVarAtom(hasCat,"B","C"),new Conjunction(getVarAtom(SpecialPredicates.Unequal,"A","B"),getVarAtom(unknown,"A"))),
													getVarAtom(intLink,"A","B")),
									getVarAtom(hasCat,"A","C"),1.0);
		model.addKernel(rules[2]);
		}
		
		
		//============= User similarity Evidence =================
		
		//##### Non-set version
//		rules[7] = new SoftRuleType(new Conjunction(new Conjunction(
//																	new Conjunction(getVarAtom(usertalk,"A","B"),getVarAtom(unknown,"D")),
//																	new Conjunction(getVarAtom(edit,"D","A"),getVarAtom(edit,"E","B"))),
//													new Conjunction(getVarAtom(known,"E"),getVarAtom(hasCat,"E","C"))),
//									getVarAtom(hasCat,"D","C"),5.0);
//		model.addModelEvidence(rules[7]);
//		
//		
//		
//		rules[8] = new SoftRuleType(new Conjunction(new Conjunction(
//																	new Conjunction(getVarAtom(usertalk,"A","B"),getVarAtom(unknown,"D")),
//																	new Conjunction(getVarAtom(edit,"D","A"),getVarAtom(edit,"E","B"))),
//													new Conjunction(new Conjunction(getVarAtom(known,"E"),getVarAtom(SpecialPredicates.NonSymmetric,"D","E")),
//																	getVarAtom(hasCat,"E","C"))),
//									getVarAtom(hasCat,"D","C"),5.0);
//		model.addModelEvidence(rules[8]);
		
		//#### Set version no 1
//		SetTerm t4a = new FormulaSetTerm(getVarAtom(edit,"D","U"),new Variable("D"),ImmutableSet.of(new Variable("U")));
//		SetTerm t4b = new VariableSetTerm(new Variable("C"),ArgumentType.Entity);
//		SetEntityDefinitionType set4 = new SetEntityDefinitionType(userCategory,t4a,t4b,new Variable[]{new Variable("U"), new Variable("C")},hasCat,"setaverage");
//		model.addModelEvidence(set4);
//		
//		
//		rules[9] = new SoftRuleType(new Conjunction(getVarAtom(edit,"D","A"),new Conjunction(getVarAtom(userCategory,"A","C"),getVarAtom(unknown,"D"))),
//									getVarAtom(hasCat,"D","C"),5.0);
//		model.addModelEvidence(rules[9]);
		
		//#### Set version no. 2
//		SetTerm t5a = new FormulaSetTerm(new Conjunction(new Conjunction(getVarAtom(edit,"D","U"),getVarAtom(unknown,"D")),
//														new Conjunction(getVarAtom(SpecialPredicates.Unequal,"D","E"),getVarAtom(edit,"E","U"))),
//										new Variable("E"),ImmutableSet.of(new Variable("D")));
//		SetTerm t5b = new VariableSetTerm(new Variable("C"),ArgumentType.Entity);
//		SetEntityDefinitionType set5 = new SetEntityDefinitionType(editCategory,t5a,t5b,new Variable[]{new Variable("D"), new Variable("C")},hasCat,"setaverage",true);
//		model.addModelEvidence(set5);
//		
//		rules[9] = new SoftRuleType(getVarAtom(editCategory,"A","C"),
//									getVarAtom(hasCat,"A","C"),0.151);
//		model.addModelEvidence(rules[9]);
		
//		SetTerm t6a = new FormulaSetTerm(new Conjunction(new Conjunction(getVarAtom(talk,"D","U"),getVarAtom(unknown,"E")),
//														new Conjunction(getVarAtom(SpecialPredicates.Unequal,"D","E"),getVarAtom(talk,"E","U"))),
//										new Variable("E"),ImmutableSet.of(new Variable("D")));
//		SetTerm t6b = new VariableSetTerm(new Variable("C"),ArgumentType.Entity);
//		SetEntityDefinitionType set6 = new SetEntityDefinitionType(talkCategory,t6a,t6b,new Variable[]{new Variable("D"), new Variable("C")},hasCat,new AggregateSetAverage(1.0));
//		model.addModelEvidence(set6);
//		
//		rules[10] = new SoftRuleType(getVarAtom(hasCat,"A","C"),
//									getVarAtom(talkCategory,"A","C"),1.0);
//		model.addModelEvidence(rules[10]);

		
		
		//#### Direct edit comparison version
//		rules[9] = new SoftRuleType(new Conjunction(new Conjunction(getVarAtom(edit,"D","A"),getVarAtom(edit,"E","A")),
//													new Conjunction(getVarAtom(hasCat,"E","C"),new Conjunction(getVarAtom(SpecialPredicates.Unequal,"D","E"),getVarAtom(unknown,"D")))),
//		getVarAtom(hasCat,"D","C"),1.0);
//		model.addModelEvidence(rules[9]);
	
		
		
	
		if (mode==Mode.AT || mode==Mode.ALT) {
		rules[10] = new SoftRuleKernel(new Conjunction(new Conjunction(getVarAtom(talk,"D","A"),getVarAtom(talk,"E","A")),
													 new Conjunction(getVarAtom(hasCat,"E","C"),new Conjunction(getVarAtom(SpecialPredicates.Unequal,"D","E"),getVarAtom(unknown,"D")))),
		getVarAtom(hasCat,"D","C"),1.0);
		model.addKernel(rules[10]);
		}

		

		//============= Basic constraints =================
		
//		PriorWeightType prior1 = new PriorWeightType(hasCat,0.2);
//		model.addModelEvidence(prior1);
		
		PredicateConstraintKernel c1 = new PredicateConstraintKernel(hasCat,PredicateConstraintType.PartialFunctional);
		model.addKernel(c1);
		
		
	}
	
	@Override
	public Measure precisionRecall(int writeID, int dataID, int truthID) {
		String[] var = {"res"};
		double[] numCorrect = queryStats("SELECT count(h.doc) as res FROM categorybelonging h, categorybelonging k WHERE " +
				" h.fold="+writeID+" and h.truth=(select max(truth) from categorybelonging as f where f.doc=h.doc and f.fold="+writeID+") " +
				"and k.fold="+truthID+" and k.doc=h.doc and k.category=h.category",
				var);
		double[] count = queryStats("SELECT COUNT(DISTINCT c.doc) as res from unknown c where c.fold="+dataID,var);
		double[] numPredicted = queryStats("SELECT count(DISTINCT h.doc) as res FROM categorybelonging h WHERE " +
				" h.fold="+writeID, var);
		log.debug("Num recalled {} of {}",numPredicted[0],count[0]);
		log.debug("Num correct {} of {}",numCorrect[0],numPredicted[0]);
		return new Measure(count[0],numCorrect[0],numPredicted[0]);
	}

}
