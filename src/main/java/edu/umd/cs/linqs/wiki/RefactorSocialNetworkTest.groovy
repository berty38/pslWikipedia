package edu.umd.cs.linqs.wiki;

import org.slf4j.Logger
import org.slf4j.LoggerFactory

import edu.emory.mathcs.utils.ConcurrencyUtils
import edu.umd.cs.psl.application.inference.MPEInference
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
import edu.umd.cs.psl.groovy.*
import edu.umd.cs.psl.model.argument.ArgumentType
import edu.umd.cs.psl.model.argument.GroundTerm
import edu.umd.cs.psl.model.argument.Variable
import edu.umd.cs.psl.model.atom.QueryAtom
import edu.umd.cs.psl.model.function.AttributeSimilarityFunction
import edu.umd.cs.psl.ui.loading.*

Logger log = LoggerFactory.getLogger(this.class);

class PartyAffilication implements AttributeSimilarityFunction {
	
	@Override
	public double similarity(String a, String b) {
		return (a.charAt(0)==b.charAt(0)?1.0:0.0);
	}
	
}

class PersonalBias implements AttributeSimilarityFunction {
	
	@Override
	public double similarity(String a, String b) {
		double value = Double.parseDouble(a);
		if (b.charAt(0)=='R') {
			if (value>=0) return 0.0;
			else return Math.abs(value)
		} else if (b.charAt(0)=='D') {
			if (value<=0) return 0.0;
			else return value;
		} else throw new IllegalArgumentException();
	}
	
}

ConfigManager cm = ConfigManager.getManager();
ConfigBundle snBundle = cm.getBundle("");
ConcurrencyUtils.setNumberOfThreads(1);

String dbpath = (args.length > 1) ? snBundle.getString("dbpath", "./") + args[1] + "/" : snBundle.getString("dbpath", "./");
DataStore data = new RDBMSDataStore(new H2DatabaseDriver(Type.Disk, dbpath, true), snBundle);

PSLModel m = new PSLModel(this, data);

m.add predicate: "knows" , types: [ArgumentType.UniqueID, ArgumentType.UniqueID]
m.add predicate: "knowswell" , types: [ArgumentType.UniqueID, ArgumentType.UniqueID]
m.add predicate: "mentor" , types: [ArgumentType.UniqueID, ArgumentType.UniqueID]
m.add predicate: "boss" , types: [ArgumentType.UniqueID, ArgumentType.UniqueID]
m.add predicate: "olderRelative" , types: [ArgumentType.UniqueID, ArgumentType.UniqueID]
m.add predicate: "idol" , types: [ArgumentType.UniqueID, ArgumentType.UniqueID]
m.add predicate: "registeredAs" , types: [ArgumentType.UniqueID, ArgumentType.String]
m.add predicate: "party" , types: [ArgumentType.UniqueID, ArgumentType.String]
m.add predicate: "votes", types: [ArgumentType.UniqueID, ArgumentType.UniqueID]

m.add function: "bias" , implementation: new PersonalBias() 

m.add rule : ( registeredAs(A,X) & party(P,Y) & bias(X,Y) ) >> votes(A,P),  weight : 0.5
m.add rule : ( votes(A,P) & knowswell(B,A) ) >> votes(B,P),  weight : 0.3
m.add rule : ( votes(A,P) & knows(B,A) ) >> votes(B,P),  weight : 0.1
m.add rule : ( votes(A,P) & boss(B,A) ) >> votes(B,P),  weight : 0.05
m.add rule : ( votes(A,P) & mentor(B,A) ) >> votes(B,P),  weight : 0.1
m.add rule : ( votes(A,P) & olderRelative(B,A) ) >> votes(B,P),  weight : 0.7
m.add rule : ( votes(A,P) & idol(B,A) ) >> votes(B,P),  weight : 0.8

m.add PredicateConstraint.PartialFunctional , on : votes

//println m
//data.registerPredicate(knows);
//data.registerPredicate(knowswell);
//data.registerPredicate(mentor);
//data.registerPredicate(boss);
//data.registerPredicate(olDERRELatiVE);
//data.registerPredicate(IdOl);
//data.registerPredicate(registeredAs);
//data.registerPredicate(party);
//data.registerPredicate(votes);

Partition read =  new Partition(1);
Partition write = new Partition(2);

def inserter = data.getInserter(party, read);
inserter.insert 0, 'Republican';
inserter.insert 1, 'Democratic';

def path = "./data/socialNets/original/";

def file = path + snBundle.getString("data", "");

InserterLookupMap inserterLookup = new InserterLookupMap();
inserterLookup.put "knows", data.getInserter(knows, read);
inserterLookup.put "knowswell", data.getInserter(knowswell, read);
inserterLookup.put "mentor", data.getInserter(mentor, read);
inserterLookup.put "boss", data.getInserter(boss, read);
inserterLookup.put "olderRelative", data.getInserter(olderRelative, read);
inserterLookup.put "idol", data.getInserter(idol, read);
inserterLookup.put "anon1", data.getInserter(registeredAs, read);
InserterUtils.loadDelimitedMultiData inserterLookup, 1, file

/* Activates everything */
//int numEntities = data.getDatabase().query(registeredAs(X,Y).getFormula()).size();
//
//int numKnows = data.getDatabase().query(knows(X,Y).getFormula()).size();
//int numKnowsWell = data.getDatabase().query(knowswell(X,Y).getFormula()).size();
//int numMentor = data.getDatabase().query(mentor(X,Y).getFormula()).size();
//int numBoss = data.getDatabase().query(boss(X,Y).getFormula()).size();
//int numOlderRelative = data.getDatabase().query(olderRelative(X,Y).getFormula()).size();
//int numIdol = data.getDatabase().query(idol(X,Y).getFormula()).size();
//
//int numLinks = numKnows + numKnowsWell + numMentor + numBoss + numOlderRelative + numIdol;
//println "Num entities: " + numEntities;
//println "Num links: " + numLinks;
//
//int total = 0;
//int unaryTotal = 0;
//int constraintTotal = 0;
//int linkTotal = 0;
//for (GroundKernel gk : app.getGroundKernel()) {
//	total++;
//	if (gk instanceof GroundConstraintKernel)
//		constraintTotal++;
//	else if (gk instanceof GroundCompatibilityKernel) {
//		if (((GroundCompatibilityKernel) gk).getWeight().getWeight() == 0.5)
//			unaryTotal++;
//		else
//			linkTotal++;
//	}
//	else
//		throw new IllegalStateException();
//}
//println "Total ground kernels: " + total;
//println "Total constraint ground kernels: " + constraintTotal;
//println "Total unary ground kernels: " + unaryTotal;
//println "Total link ground kernels: " + linkTotal;


Database db = data.getDatabase(write, read);
int numEntities = db.executeQuery(new DatabaseQuery(registeredAs(X,Y).getFormula())).size();

DatabasePopulator dbPop = new DatabasePopulator(db);
Variable Party = new Variable("Party");
Set<GroundTerm> partyGroundings = new HashSet<GroundTerm>();
partyGroundings.add(data.getUniqueID(0));
partyGroundings.add(data.getUniqueID(1));

Variable Person = new Variable("Person");
Set<GroundTerm> personGroundings = new HashSet<GroundTerm>();
for (int i = 0; i < numEntities; i++)
	personGroundings.add(data.getUniqueID(i));

Map<Variable, Set<GroundTerm>> substitutions = new HashMap<Variable, Set<GroundTerm>>();
substitutions.put(Person, personGroundings);
substitutions.put(Party, partyGroundings);
dbPop.populate(new QueryAtom(votes, Person, Party), substitutions);
//dbPop.populate((votes(Person.toAtomVariable(), PartyId.toAtomVariable())).getFormula(), substitutions);

/* Runs inference */
MPEInference mpe = new MPEInference(m, db, snBundle);
FullInferenceResult result = mpe.mpeInference();

System.out.println("Objective: " + result.getTotalIncompatibility());

/* We close the Database to make sure all writes are flushed */
db.close();

//log.debug("Computing infeasibility.");
//
//double totalInf = 0.0;
//double inf, value;
//for (GroundKernel gk : app.getGroundKernel()) {
//	if (gk instanceof GroundConstraintKernel) {
//		ConstraintTerm con = ((GroundConstraintKernel) gk).getConstraintDefinition();
//		value = con.getFunction().getValue();
//		if ((FunctionComparator.SmallerThan.equals(con.getComparator()) && value > con.getValue())
//				|| (FunctionComparator.Equality.equals(con.getComparator()))
//				|| (FunctionComparator.LargerThan.equals(con.getComparator()) && value < con.getValue())) {
//			inf = value - con.getValue();
//			totalInf += inf * inf;
//		}
//	}
//}
//totalInf = Math.sqrt(totalInf);
//log.debug("Total infeasibility: {}", totalInf);
//
//log.debug("Repairing feasibility.");
//
//Atom a, b;
//for (GroundKernel gk : app.getGroundKernel()) {
//	a = null;
//	b = null;
//	if (gk instanceof GroundConstraintKernel) {
//		ConstraintTerm con = ((GroundConstraintKernel) gk).getConstraintDefinition();
//		value = con.getFunction().getValue();
//		if ((FunctionComparator.SmallerThan.equals(con.getComparator()) && value > con.getValue())
//				|| (FunctionComparator.Equality.equals(con.getComparator()))
//				|| (FunctionComparator.LargerThan.equals(con.getComparator()) && value < con.getValue())) {
//			if (gk.getAtoms().size() == 2 && FunctionComparator.SmallerThan.equals(con.getComparator())) {
//				diff = (con.getValue() - con.getFunction().getValue()) / 2;
//				/* Pulls out the atoms */
//				for (Atom atom : gk.getAtoms()){
//					atom.setSoftValue(0, atom.getSoftValue(0) + diff);
//				}
//			}
//			else
//				throw new IllegalStateException("Only repairs less than constraints with two atoms.");
//		}
//	}
//}
//
//log.debug("Finished repairing feasibility.");
//
//log.debug("Computing infeasibility.");
//
//totalInf = 0.0;
//for (GroundKernel gk : app.getGroundKernel()) {
//	if (gk instanceof GroundConstraintKernel) {
//		ConstraintTerm con = ((GroundConstraintKernel) gk).getConstraintDefinition();
//		value = con.getFunction().getValue();
//		if ((FunctionComparator.SmallerThan.equals(con.getComparator()) && value > con.getValue())
//				|| (FunctionComparator.Equality.equals(con.getComparator()))
//				|| (FunctionComparator.LargerThan.equals(con.getComparator()) && value < con.getValue())) {
//			inf = value - con.getValue();
//			totalInf += inf * inf;
//		}
//	}
//}
//totalInf = Math.sqrt(totalInf);
//log.debug("Total infeasibility: {}", totalInf);
//
//println "Total incompatbility: " + app.getTotalIncompatibility();
