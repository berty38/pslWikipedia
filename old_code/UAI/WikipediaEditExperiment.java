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

public class WikipediaEditExperiment extends AbstractWikipediaExperiment {
	
	StandardPredicate linkOverlap, talkOverlap, textOverlap, categoryOverlap, infEdit, infTalk;
	
	StandardPredicate inferedEvent, knownEvent;
	String inferedEventTable;

	
	public WikipediaEditExperiment(Connection db) {
		super(db);
		predicatesToClose = new HashSet<Predicate>();
	}
	
	@Override
	public void loadSchema() {
		registerPredicate(new RDBMSPredicateHandle(document, "document", new String[]{"doc", "text", "quality"}, true, "fold", null,null, null),
				RDBMSArgumentTypes.LongString);
		registerPredicate(new RDBMSPredicateHandle(similarText, "similartext", new String[]{"doc1", "doc2"}, true, "fold", "truth", null,null));
		registerPredicate(new RDBMSPredicateHandle(category, "category", new String[]{"category", "name"}, true, "fold", null, null,null));
		registerPredicate(new RDBMSPredicateHandle(intLink, "withinwithinlink", new String[]{"fromDoc", "toDoc"}, true, "fold", null,null, null));
		registerPredicate(new RDBMSPredicateHandle(edit, "editevent", new String[]{"doc", "user"}, true, "fold", "truth",null, "psl"));
		registerPredicate(new RDBMSPredicateHandle(talk, "talkevent", new String[]{"doc", "user"}, true, "fold", "truth", null,"psl"));
		registerPredicate(new RDBMSPredicateHandle(infEdit, "infedit", new String[]{"doc", "user"}, false, "fold", "truth",null, "psl"));
		registerPredicate(new RDBMSPredicateHandle(infTalk, "inftalk", new String[]{"doc", "user"}, false, "fold", "truth",null, "psl"));
		registerPredicate(new RDBMSPredicateHandle(usertalk, "usertalk", new String[]{"user1", "user2"}, true, "fold", null,null, null));
		registerPredicate(new RDBMSPredicateHandle(unknown, "unknown", new String[]{"doc"}, true, "fold", null,null, null));
		registerPredicate(new RDBMSPredicateHandle(known, "known", new String[]{"doc"}, true, "fold", null,null, null));
		registerPredicate(new RDBMSPredicateHandle(hasCat, "categorybelonging", new String[]{"doc", "category"}, true, "fold", null, null,null));
		registerPredicate(new RDBMSPredicateHandle(talkOverlap, "talkoverlap", new String[]{"user"}, false, "fold", "truth", null,"psl"));
		registerPredicate(new RDBMSPredicateHandle(linkOverlap, "linkoverlap", new String[]{"doc","user"}, false, "fold", "truth", null,"psl"));
		registerPredicate(new RDBMSPredicateHandle(textOverlap, "textoverlap", new String[]{"doc"}, false, "fold", "truth",null, "psl"));
		registerPredicate(new RDBMSPredicateHandle(categoryOverlap, "categoryoverlap", new String[]{"doc"}, false, "fold", "truth",null, "psl"));
		inferedEventTable = "infedit";
	}

	@Override
	public void loadModel(Model model, PredicateFactory predFac) {
		super.loadModel(model, predFac);
		talkOverlap = predFac.createStandardPredicate("talkoverlap", new ArgumentType[]{ArgumentType.Entity});
		linkOverlap = predFac.createStandardPredicate("linkoverlap", new ArgumentType[]{ArgumentType.Entity,ArgumentType.Entity});
		textOverlap = predFac.createStandardPredicate("textoverlap",	new ArgumentType[]{ArgumentType.Entity});
		categoryOverlap = predFac.createStandardPredicate("categoryoverlap",	new ArgumentType[]{ArgumentType.Entity});
		infEdit = predFac.createStandardPredicate("infedit", new ArgumentType[]{ArgumentType.Entity, ArgumentType.Entity});
		infTalk = predFac.createStandardPredicate("inftalk", new ArgumentType[]{ArgumentType.Entity, ArgumentType.Entity});
		
		inferedEvent = infEdit;
		knownEvent = edit;
		
		//Evidence
		predicatesToClose.add(inferedEvent);
		rules = new SoftRuleKernel[20];
		
		
		rules[10] = new SoftRuleKernel(new Conjunction(getVarAtom(knownEvent,"D","U"),getVarAtom(intLink,"D","E")),
				getVarAtom(inferedEvent,"E","U"),1.0);
		model.addKernel(rules[10]);
		
//		rules[12] = new SoftRuleType(new Conjunction(getVarAtom(knownEvent,"D","U"),getVarAtom(intLink,"E","D")),
//				getVarAtom(inferedEvent,"E","U"),1.0);
//		model.addModelEvidence(rules[12]);
		
//		rules[11] = new SoftRuleType(new Conjunction(getVarAtom(knownEvent,"D","U"),getVarAtom(similarText,"D","E")),
//				getVarAtom(inferedEvent,"E","U"),1.0);
//		model.addModelEvidence(rules[11]);
//		
//		rules[13] = new SoftRuleType(new Conjunction(
//											new Conjunction(getVarAtom(knownEvent,"D","U"),getVarAtom(intLink,"D","E")),
//											new Conjunction(getVarAtom(hasCat,"D","C"),getVarAtom(hasCat,"E","C"))),
//				getVarAtom(inferedEvent,"E","U"),1.0);
//		model.addModelEvidence(rules[13]);
//		
//		rules[14] = new SoftRuleType(new Conjunction(
//				new Conjunction(getVarAtom(knownEvent,"D","U"),getVarAtom(intLink,"E","D")),
//				new Conjunction(getVarAtom(hasCat,"D","C"),getVarAtom(hasCat,"E","C"))),
//				getVarAtom(inferedEvent,"E","U"),1.0);
//		model.addModelEvidence(rules[14]);
//		
//		rules[15] = new SoftRuleType(new Conjunction(
//				new Conjunction(getVarAtom(knownEvent,"D","U"),getVarAtom(similarText,"D","E")),
//				new Conjunction(getVarAtom(hasCat,"D","C"),getVarAtom(hasCat,"E","C"))),
//				getVarAtom(inferedEvent,"E","U"),1.0);
//		model.addModelEvidence(rules[15]);
		
		
//		SetTerm t1a1 = new FormulaSetTerm(getVarAtom(intLink,"D","X"),new Variable("X"),ImmutableSet.of(new Variable("D")));
//		SetTerm t1a2 = new FormulaSetTerm(getVarAtom(intLink,"X","D"),new Variable("X"),ImmutableSet.of(new Variable("D")));
//		SetTerm t1a3 = new FormulaSetTerm(getVarAtom(similarText,"D","X"),new Variable("X"),ImmutableSet.of(new Variable("D")));
//		SetTerm t1aa = new SetUnion(t1a1,t1a2);
//		SetTerm t1a = new SetUnion(t1aa,t1a3);
//		SetTerm t1b = new VariableSetTerm(new Variable("U"),ArgumentType.Entity);
//		SetEntityDefinitionType set1 = new SetEntityDefinitionType(linkOverlap,t1a1,t1b,new Variable[]{new Variable("D"),new Variable("U")},knownEvent,"setconstant");
//		model.addModelEvidence(set1);
		
//		rules[1] = new SoftRuleType(getVarAtom(inferedEvent,"D","U"),
//									getVarAtom(linkOverlap,"D","U"),2.5);
//		model.addModelEvidence(rules[1]);
		
		
		//##Set Version
//		SetTerm t1a1 = new FormulaSetTerm(getVarAtom(intLink,"A","X"),new Variable("X"),ImmutableSet.of(new Variable("A")));
//		SetTerm t1a2 = new FormulaSetTerm(getVarAtom(intLink,"X","A"),new Variable("X"),ImmutableSet.of(new Variable("A")));
//		SetTerm t1a = new SetUnion(t1a1,t1a2);
//		SetTerm t1b = new FormulaSetTerm(getVarAtom(knownEvent,"A","U"),new Variable("U"),ImmutableSet.of(new Variable("A")));
//		SetEntityDefinitionType set1 = new SetEntityDefinitionType(linkOverlap,t1a,t1b,new Variable[]{new Variable("A")},inferedEvent,"setoverlap");
//		model.addModelEvidence(set1);
//		
//		rules[1] = new SoftRuleType(getVarAtom(known,"A"),
//									getVarAtom(linkOverlap,"A"),1.0);
//		model.addModelEvidence(rules[1]);
//		
//		SetTerm t2a = new FormulaSetTerm(getVarAtom(similarText,"A","X"),new Variable("X"),ImmutableSet.of(new Variable("A")));
//		SetTerm t2b = new FormulaSetTerm(getVarAtom(knownEvent,"A","U"),new Variable("U"),ImmutableSet.of(new Variable("A")));
//		SetEntityDefinitionType set2 = new SetEntityDefinitionType(textOverlap,t2a,t2b,new Variable[]{new Variable("A")},inferedEvent,"setoverlap");
//		model.addModelEvidence(set2);
//		
//		rules[2] = new SoftRuleType(getVarAtom(known,"A"),
//									getVarAtom(textOverlap,"A"),1.0);
//		model.addModelEvidence(rules[2]);
//		
//		SetTerm t3a = new FormulaSetTerm(new Conjunction(getVarAtom(hasCat,"A","Y"),getVarAtom(hasCat,"X","Y")),new Variable("X"),ImmutableSet.of(new Variable("A")));
//		SetTerm t3b = new FormulaSetTerm(getVarAtom(knownEvent,"A","U"),new Variable("U"),ImmutableSet.of(new Variable("A")));
//		SetEntityDefinitionType set3 = new SetEntityDefinitionType(categoryOverlap,t3a,t3b,new Variable[]{new Variable("A")},inferedEvent,"setoverlap");
//		model.addModelEvidence(set3);
//		
//		rules[3] = new SoftRuleType(getVarAtom(known,"A"),
//									getVarAtom(categoryOverlap,"A"),1.0);
//		model.addModelEvidence(rules[3]);
		
		
		
		//============= Basic constraints =================
		
//		PriorWeightType prior1 = new PriorWeightType(inferedEvent,0.4);
//		model.addModelEvidence(prior1);
		
//		FunctionalConstraintsType c1 = new FunctionalConstraintsType(hasCat,FunctionalConstraintsType.Type.PartialFunctional);
//		model.addModelEvidence(c1);
		
		
	}
	
	@Override
	public Measure precisionRecall(int writeID, int dataID, int truthID) {
		double bestf1 = 0.0;
		Measure bestMeasure = null;
		for (double threshold=0.2;threshold<=1.0;threshold+=0.2) {
			String[] var = {"res"};
			double[] numCorrect = queryStats("SELECT count(h.*) as res FROM "+inferedEventTable+" h, "+inferedEventTable+" k WHERE " +
					" h.fold="+writeID+" and h.truth>="+threshold+" and h.doc=k.doc and h.user=k.user"+
					" and k.fold="+truthID,
					var);
			double[] count = queryStats("SELECT COUNT(h.*) as res from "+inferedEventTable+" h where h.fold="+truthID,var);
			double[] numPredicted = queryStats("SELECT COUNT(h.*) as res from "+inferedEventTable+" h where h.fold="+writeID+
					" and h.truth>="+threshold,var);
			Measure m = new Measure(count[0],numCorrect[0],numPredicted[0]);

			log.debug("Num correct {} of {}",numCorrect[0],numPredicted[0]);
			log.debug("Num recalled {} of {}",numCorrect[0],count[0]);
			log.debug("F1 measure {}",m.f1());
			if (m.f1()>bestf1) {
				bestMeasure = m;
				bestf1 = m.f1();
			}
		}
		return bestMeasure;
	}


}
