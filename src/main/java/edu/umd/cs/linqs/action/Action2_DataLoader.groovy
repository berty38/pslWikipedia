package edu.umd.cs.linqs.action

import org.slf4j.Logger
import org.slf4j.LoggerFactory

import edu.umd.cs.psl.config.*
import edu.umd.cs.psl.database.DataStore
import edu.umd.cs.psl.database.Partition
import edu.umd.cs.psl.database.loading.Inserter
import edu.umd.cs.psl.database.rdbms.RDBMSDataStore
import edu.umd.cs.psl.database.rdbms.driver.H2DatabaseDriver
import edu.umd.cs.psl.database.rdbms.driver.H2DatabaseDriver.Type
import edu.umd.cs.psl.groovy.*
import edu.umd.cs.psl.model.argument.ArgumentType
import edu.umd.cs.psl.ui.loading.*


/*** CONFIGURATION PARAMETERS ***/

Logger log = LoggerFactory.getLogger(this.class)
ConfigManager cm = ConfigManager.getManager();
ConfigBundle cb = cm.getBundle("action");

//def defPath = "data/action/action";
def defPath = System.getProperty("java.io.tmpdir") + "/action2"
def dbpath = cb.getString("dbpath", defPath)
DataStore data = new RDBMSDataStore(new H2DatabaseDriver(Type.Disk, dbpath, true), cb)

int numSeqs = 63;


/*** KB DEFINITION ***/

log.info("Initializing KB ...");

PSLModel m = new PSLModel(this, data);

/* PREDICATES */

// target
m.add predicate: "doing", types: [ArgumentType.UniqueID,ArgumentType.Integer];
m.add predicate: "sameObj", types: [ArgumentType.UniqueID,ArgumentType.UniqueID];

// observed
m.add predicate: "inFrame", types: [ArgumentType.UniqueID,ArgumentType.Integer,ArgumentType.Integer];
m.add predicate: "inSameFrame", types: [ArgumentType.UniqueID,ArgumentType.UniqueID];
m.add predicate: "inSeqFrames", types: [ArgumentType.UniqueID,ArgumentType.UniqueID];
m.add predicate: "dims", types: [ArgumentType.UniqueID,ArgumentType.Integer,ArgumentType.Integer,ArgumentType.Integer,ArgumentType.Integer];
m.add predicate: "acdAction", types: [ArgumentType.UniqueID,ArgumentType.Integer];
//m.add predicate: "hogAction", types: [ArgumentType.UniqueID,ArgumentType.Integer];
//m.add predicate: "nhogScore", types: [ArgumentType.UniqueID,ArgumentType.Integer];


/** DATASTORE PARTITIONS **/

int partCnt = 0;
Partition[][] partitions = new Partition[2][numSeqs];
for (int s = 0; s < numSeqs; s++) {
	partitions[0][s] = new Partition(partCnt++);	// observations
	partitions[1][s] = new Partition(partCnt++);	// labels
}
	
	
/*** LOAD DATA ***/

log.info("Loading data ...");

def dataPath = "./data/action/"
def filePfx = dataPath + "d2_";

Inserter[] inserters;

/* Ground truth */
inserters = InserterUtils.getMultiPartitionInserters(data, doing, partitions[1], numSeqs);
InserterUtils.loadDelimitedDataMultiPartition(inserters, filePfx + "action.txt");
inserters = InserterUtils.getMultiPartitionInserters(data, sameObj, partitions[1], numSeqs);
InserterUtils.loadDelimitedDataMultiPartition(inserters, filePfx + "sameobj.txt");

/* Observations */
inserters = InserterUtils.getMultiPartitionInserters(data, inFrame, partitions[0], numSeqs);
InserterUtils.loadDelimitedDataMultiPartition(inserters, filePfx + "inframe.txt");
inserters = InserterUtils.getMultiPartitionInserters(data, inSameFrame, partitions[0], numSeqs);
InserterUtils.loadDelimitedDataMultiPartition(inserters, filePfx + "insameframe.txt");
inserters = InserterUtils.getMultiPartitionInserters(data, inSeqFrames, partitions[0], numSeqs);
InserterUtils.loadDelimitedDataMultiPartition(inserters, filePfx + "inseqframes.txt");
inserters = InserterUtils.getMultiPartitionInserters(data, dims, partitions[0], numSeqs);
InserterUtils.loadDelimitedDataMultiPartition(inserters, filePfx + "coords.txt");
inserters = InserterUtils.getMultiPartitionInserters(data, acdAction, partitions[0], numSeqs);
InserterUtils.loadDelimitedDataTruthMultiPartition(inserters, filePfx + "acdaction.txt");
//inserters = InserterUtils.getMultiPartitionInserters(data, hogAction, partitions[0], numSeqs);
//InserterUtils.loadDelimitedDataTruthMultiPartition(inserters, filePfx + "hogaction.txt");
//inserters = InserterUtils.getMultiPartitionInserters(data, nhogScore, partitions[0], numSeqs);
//InserterUtils.loadDelimitedDataTruthMultiPartition(inserters, filePfx + "nhogscores.txt");

log.info("Done!");
