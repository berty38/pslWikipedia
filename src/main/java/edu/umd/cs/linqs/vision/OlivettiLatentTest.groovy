package edu.umd.cs.linqs.vision

import org.slf4j.Logger
import org.slf4j.LoggerFactory

import edu.umd.cs.linqs.vision.PatchStructure.Patch
import edu.umd.cs.linqs.wiki.DataOutputter
import edu.umd.cs.psl.application.inference.MPEInference
import edu.umd.cs.psl.application.learning.weight.WeightLearningApplication
import edu.umd.cs.psl.application.learning.weight.maxlikelihood.MaxLikelihoodMPE
import edu.umd.cs.psl.application.learning.weight.maxlikelihood.MaxPseudoLikelihood
import edu.umd.cs.psl.config.ConfigBundle
import edu.umd.cs.psl.config.ConfigManager
import edu.umd.cs.psl.database.DataStore
import edu.umd.cs.psl.database.Database
import edu.umd.cs.psl.database.Partition
import edu.umd.cs.psl.database.loading.Inserter
import edu.umd.cs.psl.database.rdbms.RDBMSDataStore
import edu.umd.cs.psl.database.rdbms.driver.H2DatabaseDriver
import edu.umd.cs.psl.database.rdbms.driver.H2DatabaseDriver.Type
import edu.umd.cs.psl.evaluation.result.FullInferenceResult
import edu.umd.cs.psl.evaluation.statistics.ContinuousPredictionComparator
import edu.umd.cs.psl.groovy.*
import edu.umd.cs.psl.model.argument.ArgumentType
import edu.umd.cs.psl.model.argument.Term
import edu.umd.cs.psl.model.argument.UniqueID
import edu.umd.cs.psl.model.argument.Variable
import edu.umd.cs.psl.model.atom.Atom
import edu.umd.cs.psl.model.atom.RandomVariableAtom
import edu.umd.cs.psl.model.predicate.Predicate
import edu.umd.cs.psl.util.database.Queries


Logger log = LoggerFactory.getLogger(this.class)

/* VISION EXPERIMENT SETTINGS */
// test on left half of face (bottom if false)
testLeft = true
// train on randomly sampled pixels
trainOnRandom = false
// number of training faces
numTraining = 3
// number of testing faces
numTesting = 3

dataset = "olivetti"


if (args.length >= 2) {
	if (args[0] == "bottom") {
		testLeft = false
		log.info("Testing on bottom of face")
	} else {
		testLeft = true
		log.info("Testing on left of face")
	}
	if (args[1] == "half") {
		trainOnRandom = false
		log.info("Training on given half of face")
	} else {
		trainOnRandom = true
		log.info("Training on randomly held-out pixels")
	}
}

def expSetup = (testLeft? "left" : "bottom") + "-" + (trainOnRandom? "rand" : "same")

ConfigManager cm = ConfigManager.getManager()
ConfigBundle config = cm.getBundle("latentfaces")

boolean sq = true

/*
 * INITIALIZES DATASTORE AND MODEL
 */
def defaultPath = System.getProperty("java.io.tmpdir") + "/"
//def defaultPath = "/scratch0/bert-uai13/"
String dbpath = config.getString("dbpath", defaultPath + "psl" + dataset)
DataStore data = new RDBMSDataStore(new H2DatabaseDriver(Type.Disk, dbpath, true), config)

PSLModel m = new PSLModel(this, data)

/*
 * DEFINE MODEL
 */

numTypes = 3

width = 64
height = 64
branching = 2
depth = 7
def hierarchy = new PatchStructure(width, height, branching, depth, config)
hierarchy.generatePixels()
//hierarchy.generateGridResolution(16)


//m.add predicate: "north", types: [ArgumentType.UniqueID, ArgumentType.UniqueID]
//m.add predicate: "east", types: [ArgumentType.UniqueID, ArgumentType.UniqueID]
//m.add predicate: "horizontalMirror", types: [ArgumentType.UniqueID, ArgumentType.UniqueID]
//m.add predicate: "verticalMirror", types: [ArgumentType.UniqueID, ArgumentType.UniqueID]
//m.add predicate: "neighbors", types: [ArgumentType.UniqueID, ArgumentType.UniqueID]
m.add predicate: "pixelBrightness", types: [ArgumentType.UniqueID, ArgumentType.UniqueID]
m.add predicate: "picture", types: [ArgumentType.UniqueID]
m.add predicate: "pictureType", types: [ArgumentType.UniqueID, ArgumentType.UniqueID]

//m.add PredicateConstraint.Functional , on : pictureType

def pictureTypes = new ArrayList<UniqueID>(numTypes)
for (int i = 0; i < numTypes; i++)
	pictureTypes.add(data.getUniqueID(i))

double initialWeight = 5.0

Random random = new Random(314159)

scale = 0.5;

for (Patch p : hierarchy.getPatches().values()) {
	UniqueID patch = data.getUniqueID(p.uniqueID())
	Variable pic = new Variable("pictureVar")
	Term [] args = new Term[2]
	args[0] = patch
	args[1] = pic

	for (Term type : pictureTypes) {
		m.add rule: (picture(pic) & pictureType(pic, type)) >> pixelBrightness(patch, pic), weight: initialWeight, squared: sq
		m.add rule: (picture(pic) & pictureType(pic, type)) >> ~pixelBrightness(patch, pic), weight: initialWeight, squared: sq
		
		m.add rule: (picture(pic) & pixelBrightness(patch, pic)) >> pictureType(pic, type), weight: initialWeight, squared: false
		m.add rule: (picture(pic) & ~pixelBrightness(patch, pic)) >> pictureType(pic, type), weight: initialWeight, squared: false
		m.add rule: (picture(pic) & pixelBrightness(patch, pic)) >> ~pictureType(pic, type), weight: initialWeight, squared: false
		m.add rule: (picture(pic) & ~pixelBrightness(patch, pic)) >> ~pictureType(pic, type), weight: initialWeight, squared: false
	}
	m.add rule: picture(pic) >> pixelBrightness(patch, pic), weight: initialWeight, squared: sq
	m.add rule: picture(pic) >> ~pixelBrightness(patch, pic), weight: initialWeight, squared: sq
}

//m.add rule : ~(pixelBrightness(X,Y)), weight: initialWeight, squared: sq
//m.add rule : ~(pictureType(X,Y)), weight: initialWeight, squared: sq

log.info("Model has {} weighted kernels", m.getKernels().size());

Partition trainObs =  new Partition(0)
Partition testObs = new Partition(1)
Partition trainLabel = new Partition(2)
Partition testLabel = new Partition(3)
Partition trainWrite = new Partition(4)
Partition testWrite = new Partition(5)
Partition trainLatent = new Partition(6)

/*
 * LOAD DATA
 */
dataDir = "data/vision"

// construct observed mask
boolean[] mask = new boolean[width * height]
boolean[] negMask = new boolean[width * height]
boolean[] trainMask = new boolean[width * height]
boolean[] negTrainMask = new boolean[width * height]
int c = 0
for (int x = 0; x < width; x++) {
	for (int y = 0; y < height; y++) {
		if (testLeft)
			mask[c] = x >= (width / 2)
		else
			mask[c] = y <= height / 2

		negMask[c] = !mask[c]

		trainMask[c] = mask[c]
		negTrainMask[c] = negMask[c]
		c++
	}
}

for (Partition part : [trainObs, testObs]) {
	def readDB = data.getDatabase(part)
	//	ImagePatchUtils.insertFromPatchMap(north, readDB, hierarchy.getNorth())
	//	ImagePatchUtils.insertFromPatchMap(east, readDB, hierarchy.getEast())
	//	ImagePatchUtils.insertFromPatchMap(horizontalMirror, readDB, hierarchy.getMirrorHorizontal())
	//	ImagePatchUtils.insertFromPatchMap(verticalMirror, readDB, hierarchy.getMirrorVertical())
	readDB.close()
}

ArrayList<double []> images = ImagePatchUtils.loadImages(dataDir + "/" + dataset + "01.txt", width, height)
// create list of train images and test images
ArrayList<double []> trainImages = new ArrayList<double[]>()
ArrayList<double []> testImages = new ArrayList<double[]>()

for (int i = 0; i < images.size(); i++) {
	if (i < numTraining) {
		trainImages.add(images.get(i))
		//	} else if (i >= images.size() - numTesting) {
		testImages.add(images.get(i))
	}
}

images.clear()

Inserter picInserter = data.getInserter(picture, trainObs)
for (int i = 0; i < trainImages.size(); i++) {
	UniqueID id = data.getUniqueID(i)
	picInserter.insert(id)
}
picInserter = data.getInserter(picture, testObs)
for (int i = 0; i < testImages.size(); i++) {
	UniqueID id = data.getUniqueID(i)
	picInserter.insert(id)
}

Random rand = new Random(0)

/** load images into base pixels **/
def trainReadDB = data.getDatabase(trainObs)
def trainLabelDB = data.getDatabase(trainLabel)
def trainWriteDB = data.getDatabase(trainWrite)
def trainLatentDB = data.getDatabase(trainLatent)
for (int i = 0; i < trainImages.size(); i++) {

	if (trainOnRandom) {
		c = 0
		for (int x = 0; x < width; x++) {
			for (int y = 0; y < height; y++) {
				trainMask[c] = rand.nextBoolean()
				negTrainMask[c] = !trainMask[c]
				c++
			}
		}
	}

	UniqueID id = data.getUniqueID(i)
	ImagePatchUtils.setObservedPixels(pixelBrightness, id, trainReadDB, hierarchy, width, height, trainImages.get(i), trainMask)
	ImagePatchUtils.setPixels(pixelBrightness, id, trainLabelDB, hierarchy, width, height, trainImages.get(i), negTrainMask)
}
populateLatentVariables(trainImages.size(), pictureType, pictureTypes, trainLatentDB, random)
populateLatentVariables(trainImages.size(), pictureType, pictureTypes, trainWriteDB, random)

trainLatentDB.close()
trainWriteDB.close()
trainReadDB.close()
trainLabelDB.close()

trainDB = data.getDatabase(trainWrite, trainObs)
/** populate open variables **/
for (int i = 0; i < trainImages.size(); i++) {
	UniqueID id = data.getUniqueID(i)
	ImagePatchUtils.populatePixels(width, height, pixelBrightness, trainDB, id)
}
trainDB.close()


def testReadDB = data.getDatabase(testObs)
def testLabelDB = data.getDatabase(testLabel)
def testWriteDB = data.getDatabase(testWrite)
for (int i = 0; i < testImages.size(); i++) {
	def id = data.getUniqueID(i)
	ImagePatchUtils.setObservedPixels(pixelBrightness, id, testReadDB, hierarchy, width, height, testImages.get(i), mask)
	ImagePatchUtils.setPixels(pixelBrightness, id, testLabelDB, hierarchy, width, height, testImages.get(i), negMask)
}
populateLatentVariables(testImages.size(), pictureType, pictureTypes, testWriteDB, random)

testWriteDB.close()
testReadDB.close()
testLabelDB.close()


testDB = data.getDatabase(testWrite, testObs)
/** populate open variables **/
for (int i = 0; i < testImages.size(); i++) {
	UniqueID id = data.getUniqueID(i)
	ImagePatchUtils.populatePixels(width, height, pixelBrightness, testDB, id)
}
testDB.close()

/** set up predicate sets **/

def eStepToClose = [picture, pixelBrightness] as Set
def mStepToClose = [picture] as Set
def labelToClose = [pixelBrightness, pictureType] as Set

/*
 * Weight learning
 */

numIterations = 10

for (int i = 0; i < numIterations; i++) {

	log.info("Starting M-step, iteration {}", i)
	// m-step: learns new weights
	trainDB = data.getDatabase(trainWrite, mStepToClose, trainObs)
	labelDB = data.getDatabase(trainLabel, labelToClose, trainLatent)
	WeightLearningApplication wl = new MaxLikelihoodMPE(m, trainDB, labelDB, config)
	wl.learn()
	wl.close()
	trainDB.close()
	labelDB.close()

	log.info("Starting E-step, iteration {}", i)
	// e-step: infers picture type
	eStepDB = data.getDatabase(trainLatent, eStepToClose, trainObs, trainLabel)
	MPEInference mpe = new MPEInference(m, eStepDB, config);
	mpe.mpeInference();
	mpe.close();
	eStepDB.close();
}

log.info("Starting inference on test set")
/*
 * Inference on test set
 */
def testDB = data.getDatabase(testWrite, mStepToClose, testObs)
MPEInference mpe = new MPEInference(m, testDB, config)
FullInferenceResult result = mpe.mpeInference()


DataOutputter.outputPredicate("output/vision/latent/"+ dataset + "-" + expSetup + ".txt" , testDB, pixelBrightness, ",", true, "index,image")
testDB.close()

/*
 * Evaluation
 */
def groundTruthDB = data.getDatabase(testLabel, labelToClose)
testDB = data.getDatabase(testWrite)
def comparator = new ContinuousPredictionComparator(testDB)
comparator.setBaseline(groundTruthDB)

def metric = ContinuousPredictionComparator.Metric.MSE
comparator.setMetric(metric)
score = comparator.compare(pixelBrightness) * (255 * 255) // scale to full 256 grayscale range

log.info("Mean squared error: " + score)

// close all databases
groundTruthDB.close()
testDB.close()

System.out.println(dataset + "-" + expSetup + ", mean squared error: " + score)

// output model
DataOutputter.outputModel("output/vision/latent/"+ dataset + "-" + expSetup + "-model.txt", m)



private void populateLatentVariables(int numImages, Predicate latentVariable, Iterable<UniqueID> latentStates, Database db, Random rand) {
	for (int i = 0; i < numImages; i++) {
		UniqueID pic = db.getUniqueID(i)
		int j = 0;
		for (UniqueID type : latentStates) {
			RandomVariableAtom latentAtom = db.getAtom(latentVariable, pic, type)
//			if (j == i)
//				latentAtom.setValue(1.0)
//			else
//				latentAtom.setValue(0.0)
			latentAtom.setValue(rand.nextDouble());
			
			latentAtom.commitToDB()
			j++;
		}
	}
}


