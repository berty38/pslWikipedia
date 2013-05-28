package edu.umd.cs.linqs.vision

import org.slf4j.Logger
import org.slf4j.LoggerFactory

import com.google.common.collect.Iterables;

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
import edu.umd.cs.psl.model.kernel.CompatibilityKernel;
import edu.umd.cs.psl.model.parameters.PositiveWeight
import edu.umd.cs.psl.model.predicate.Predicate
import edu.umd.cs.psl.util.database.Queries


Logger log = LoggerFactory.getLogger(this.class)

/* VISION EXPERIMENT SETTINGS */
// test on left half of face (bottom if false)
testLeft = true
// number of training faces
numTraining = 50
// number of testing faces
numTesting = 50

dataset = "olivetti-small"

width = 32
height = 32

if (args.length >= 2) {
	if (args[0] == "bottom") {
		testLeft = false
		log.info("Testing on bottom of face")
	} else {
		testLeft = true
		log.info("Testing on left of face")
	}
}

def expSetup = (testLeft? "left" : "bottom")

// construct observed mask
boolean[] mask = new boolean[width * height]
boolean[] negMask = new boolean[width * height]
int c = 0
for (int x = 0; x < width; x++) {
	for (int y = 0; y < height; y++) {
		if (testLeft)
			mask[c] = x >= (width / 2)
		else
			mask[c] = y <= height / 2

		negMask[c] = !mask[c]
		c++
	}
}

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

numTypes = 5
numMeans = 4
variance = 0.004

branching = 2
depth = 7
// branching and depth are now unused
def hierarchy = new PatchStructure(width, height, branching, depth, config)
hierarchy.generatePixels()


m.add predicate: "mean", types: [ArgumentType.UniqueID, ArgumentType.UniqueID]
m.add predicate: "hasMean", types: [ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID]
m.add predicate: "pixelBrightness", types: [ArgumentType.UniqueID, ArgumentType.UniqueID]
m.add predicate: "picture", types: [ArgumentType.UniqueID]
m.add predicate: "pictureType", types: [ArgumentType.UniqueID, ArgumentType.UniqueID]

//m.add PredicateConstraint.Functional , on : pictureType

double initialWeight = 1.0

Random random = new Random(314159)

//scale = 0.5;

for (Patch p : hierarchy.getPatches().values()) {
	UniqueID patch = data.getUniqueID(p.uniqueID())
	Variable pic = new Variable("pictureVar")
	Term [] args = new Term[2]
	args[0] = patch
	args[1] = pic

	int patchID = patch.getInternalID() // this probably only works if we are only using pixel "patches"

	for (int j = 0; j < numTypes; j++) {
		UniqueID type = data.getUniqueID(j)
		if (mask[patchID]) {
			// if patch is in input set, add rules to reason about mean representation
			for (int i = 0; i < numMeans; i++) {
				UniqueID mean = data.getUniqueID(i);

				m.add rule: (picture(pic) & pictureType(pic, type)) >> hasMean(pic, patch, mean), weight: initialWeight, squared: false
				m.add rule: (picture(pic) & pictureType(pic, type)) >> ~hasMean(pic, patch, mean), weight: initialWeight, squared: false
				m.add rule: (picture(pic) & ~pictureType(pic, type)) >> hasMean(pic, patch, mean), weight: initialWeight, squared: false
				m.add rule: (picture(pic) & ~pictureType(pic, type)) >> ~hasMean(pic, patch, mean), weight: initialWeight, squared: false
			}
		}
		if (!mask[patchID]) {
			// if patch is in output set, add rules to infer pixel truth value from pictureType
			m.add rule: (picture(pic) & pictureType(pic, type)) >> pixelbrightness(pic, patch), weight: initialWeight, squared: sq
			m.add rule: (picture(pic) & pictureType(pic, type)) >> ~pixelbrightness(pic, patch), weight: initialWeight, squared: sq
			m.add rule: (picture(pic) & ~(pictureType(pic, type))) >> pixelbrightness(pic, patch), weight: initialWeight, squared: sq
			m.add rule: (picture(pic) & ~(pictureType(pic, type))) >> ~pixelbrightness(pic, patch), weight: initialWeight, squared: sq
		}
	}
	if (!mask[patchID]) {
		m.add rule: picture(pic) >> pixelBrightness(pic, patch), weight: initialWeight, squared: sq
		m.add rule: picture(pic) >> ~pixelBrightness(pic, patch), weight: initialWeight, squared: sq
	}
}

for (int i = 0; i < numTypes; i++) {
	UniqueID typeA = data.getUniqueID(i)
	for (int j = 0; j < numTypes; j++) {
		UniqueID typeB = data.getUniqueID(j)
		m.add rule: pictureType(P, typeA) >> pictureType(P, typeB), weight: initialWeight, squared: sq
		m.add rule: pictureType(P, typeA) >> ~pictureType(P, typeB), weight: initialWeight, squared: sq
	}
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

ArrayList<double []> images = ImagePatchUtils.loadImages(dataDir + "/" + dataset + "01.txt", width, height)
// create list of train images and test images
ArrayList<double []> trainImages = new ArrayList<double[]>()
ArrayList<double []> testImages = new ArrayList<double[]>()

for (int i = 0; i < images.size(); i++) {
	if (i < numTraining) {
		trainImages.add(images.get(i))
	} else if (i >= images.size() - numTesting) {
		testImages.add(images.get(i))
	}
}

images.clear()

Inserter picInserter = data.getInserter(picture, trainObs)
for (int i = 0; i < trainImages.size(); i++) {
	UniqueID imageID = data.getUniqueID(i)
	picInserter.insert(imageID)
}
picInserter = data.getInserter(picture, testObs)
for (int i = 0; i < testImages.size(); i++) {
	UniqueID imageID = data.getUniqueID(i)
	picInserter.insert(imageID)
}

Random rand = new Random(0)

/** load images into base pixels **/
def trainReadDB = data.getDatabase(trainObs)
def trainWriteDB = data.getDatabase(trainWrite)
def trainLatentDB = data.getDatabase(trainLatent)
for (int i = 0; i < trainImages.size(); i++) {
	UniqueID imageID = data.getUniqueID(i)
	ImagePatchUtils.setObservedHasMean(hasMean, mean, imageID, trainReadDB, width, height, numMeans, variance, trainImages.get(i), mask)
	ImagePatchUtils.setPixels(pixelBrightness, imageID, trainReadDB, hierarchy, width, height, trainImages.get(i), mask)
}
populateLatentVariables(trainImages.size(), pictureType, numTypes, trainLatentDB, random)
populateLatentVariables(trainImages.size(), pictureType, numTypes, trainWriteDB, random)
trainLatentDB.close()
trainWriteDB.close()
trainReadDB.close()

def trainLabelDB = data.getDatabase(trainLabel, trainObs)
for (int i = 0; i < trainImages.size(); i++) {
	UniqueID imageID = data.getUniqueID(i)
	ImagePatchUtils.setTargetHasMean(hasMean, mean, imageID, trainLabelDB, width, height, numMeans, variance, trainImages.get(i), negMask)
	ImagePatchUtils.setPixels(pixelBrightness, imageID, trainLabelDB, hierarchy, width, height, trainImages.get(i), negMask)
}
trainLabelDB.close()

trainDB = data.getDatabase(trainWrite, trainObs)
/** populate open variables **/
for (int i = 0; i < trainImages.size(); i++) {
	UniqueID imageID = data.getUniqueID(i)
	ImagePatchUtils.populateHasMean(width, height, numMeans, hasMean, trainDB, imageID)
	ImagePatchUtils.populatePixels(width, height, pixelbrightness, trainDB, imageID);
}
trainDB.close()


def testReadDB = data.getDatabase(testObs)
def testLabelDB = data.getDatabase(testLabel)
def testWriteDB = data.getDatabase(testWrite)
for (int i = 0; i < testImages.size(); i++) {
	def imageID = data.getUniqueID(i)
	ImagePatchUtils.setObservedHasMean(hasMean, mean, imageID, testReadDB, width, height, numMeans, variance, testImages.get(i), mask)
	ImagePatchUtils.setPixels(pixelBrightness, imageID, testReadDB, hierarchy, width, height, testImages.get(i), mask)
	ImagePatchUtils.setPixels(pixelBrightness, imageID, testLabelDB, hierarchy, width, height, testImages.get(i), negMask)
}
populateLatentVariables(testImages.size(), pictureType, numTypes, testWriteDB, random)

testWriteDB.close()
testReadDB.close()
testLabelDB.close()


testDB = data.getDatabase(testWrite, testObs)
/** populate open variables **/
for (int i = 0; i < testImages.size(); i++) {
	UniqueID imageID = data.getUniqueID(i)
	ImagePatchUtils.populateHasMean(width, height, numMeans, hasMean, testDB, imageID)
	ImagePatchUtils.populatePixels(width, height, pixelbrightness, testDB, imageID);
}
testDB.close()

/** set up predicate sets **/

def eStepToClose = [picture, hasMean, pixelBrightness] as Set
def mStepToClose = [picture, hasMean] as Set
def labelToClose = [pixelBrightness, hasMean, pictureType] as Set

/*
 * Weight learning
 */

numIterations = 10

for (int i = 0; i < numIterations; i++) {
	//	// initialize weights
	//	for (CompatibilityKernel k : Iterables.filter(m.getKernels(), CompatibilityKernel.class))
	//			k.setWeight(new PositiveWeight(initialWeight))

	log.info("Starting M-step, iteration {}", i)
	// m-step: learns new weights
	trainDB = data.getDatabase(trainWrite, mStepToClose, trainObs)
	labelDB = data.getDatabase(trainLabel, labelToClose, trainLatent)
	WeightLearningApplication wl = new MaxLikelihoodMPE(m, trainDB, labelDB, config)
	wl.learn()
	wl.close()
	
	// output intermediate reconstruction
	DataOutputter.outputPredicate("output/vision/latent/"+ dataset + "-" + expSetup + "-train.txt" , trainDB, pixelBrightness, ",", true, "index,image")
	
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

//ImagePatchUtils.decodeBrightness(hasMean, mean, pixelBrightness, picture, testDB, width, height, numMeans)

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



private void populateLatentVariables(int numImages, Predicate latentVariable, int numTypes, Database db, Random rand) {
	for (int i = 0; i < numImages; i++) {
		UniqueID pic = db.getUniqueID(i)
		for (j = 0; j < numTypes; j++) {
			UniqueID typeID = db.getUniqueID(j);
			RandomVariableAtom latentAtom = db.getAtom(latentVariable, pic, typeID)
			//			latentAtom.setValue((rand.nextBoolean())? 0.0 : 1.0);
			latentAtom.setValue(rand.nextDouble());
			latentAtom.commitToDB();
		}
	}
}


