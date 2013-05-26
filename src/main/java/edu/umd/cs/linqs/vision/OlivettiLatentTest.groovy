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
numTraining = 5
// number of testing faces
numTesting = 50

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

numTypes = 2
numMeans = 4
variance = 0.1

width = 64
height = 64
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

double initialWeight = 5.0

Random random = new Random(314159)

//scale = 0.5;

for (Patch p : hierarchy.getPatches().values()) {
	UniqueID patch = data.getUniqueID(p.uniqueID())
	Variable pic = new Variable("pictureVar")
	Term [] args = new Term[2]
	args[0] = patch
	args[1] = pic

	for (int i = 0; i < numMeans; i++) {
		UniqueID mean = data.getUniqueID(i);
		for (int j = 0; j < numTypes; j++) {
			UniqueID type = data.getUniqueID(j)

			m.add rule: (picture(pic) & pictureType(pic, type)) >> hasMean(patch, pic, mean), weight: initialWeight, squared: sq
			m.add rule: (picture(pic) & pictureType(pic, type)) >> ~hasMean(patch, pic, mean), weight: initialWeight, squared: sq

			m.add rule: (picture(pic) & hasMean(patch, pic, mean)) >> pictureType(pic, type), weight: initialWeight, squared: false
			m.add rule: (picture(pic) & ~hasMean(patch, pic, mean)) >> pictureType(pic, type), weight: initialWeight, squared: false
			m.add rule: (picture(pic) & hasMean(patch, pic, mean)) >> ~pictureType(pic, type), weight: initialWeight, squared: false
			m.add rule: (picture(pic) & ~hasMean(patch, pic, mean)) >> ~pictureType(pic, type), weight: initialWeight, squared: false
		}
		m.add rule: picture(pic) >> hasMean(patch, pic, mean), weight: initialWeight, squared: sq
		m.add rule: picture(pic) >> ~hasMean(patch, pic, mean), weight: initialWeight, squared: sq
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

// construct observed mask
boolean[] testMask = new boolean[width * height]
boolean[] negMask = new boolean[width * height]
boolean[] trainMask = new boolean[width * height]
boolean[] negTrainMask = new boolean[width * height]
int c = 0
for (int x = 0; x < width; x++) {
	for (int y = 0; y < height; y++) {
		if (testLeft)
			testMask[c] = x >= (width / 2)
		else
			testMask[c] = y <= height / 2

		negMask[c] = !testMask[c]

		trainMask[c] = testMask[c]
		negTrainMask[c] = negMask[c]
		c++
	}
}


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

	UniqueID imageID = data.getUniqueID(i)
	ImagePatchUtils.setObservedHasMean(hasMean, mean, imageID, trainReadDB, width, height, numMeans, variance, trainImages.get(i), trainMask)
	ImagePatchUtils.setPixels(pixelBrightness, imageID, trainReadDB, hierarchy, width, height, trainImages.get(i), trainMask)
	ImagePatchUtils.setObservedHasMean(hasMean, mean, imageID, trainLabelDB, width, height, numMeans, variance, trainImages.get(i), negTrainMask)
	ImagePatchUtils.setPixels(pixelBrightness, imageID, trainLabelDB, hierarchy, width, height, trainImages.get(i), negTrainMask)
}
populateLatentVariables(trainImages.size(), pictureType, numTypes, trainLatentDB, random)
populateLatentVariables(trainImages.size(), pictureType, numTypes, trainWriteDB, random)

trainLatentDB.close()
trainWriteDB.close()
trainReadDB.close()
trainLabelDB.close()

trainDB = data.getDatabase(trainWrite, trainObs)
/** populate open variables **/
for (int i = 0; i < trainImages.size(); i++) {
	UniqueID imageID = data.getUniqueID(i)
	ImagePatchUtils.populateHasMean(width, height, numMeans, hasMean, trainDB, imageID)
}
trainDB.close()


def testReadDB = data.getDatabase(testObs)
def testLabelDB = data.getDatabase(testLabel)
def testWriteDB = data.getDatabase(testWrite)
for (int i = 0; i < testImages.size(); i++) {
	def imageID = data.getUniqueID(i)
	ImagePatchUtils.setObservedHasMean(hasMean, mean, imageID, testReadDB, width, height, numMeans, variance, testImages.get(i), testMask)
	ImagePatchUtils.setPixels(pixelBrightness, imageID, testReadDB, hierarchy, width, height, testImages.get(i), testMask)
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

ImagePatchUtils.decodeBrightness(hasMean, mean, pixelBrightness, picture, testDB, width, height, numMeans)

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
		int j = 0;
		for (j = 0; j < numTypes; j++) {
			UniqueID typeID = db.getUniqueID(j);
			RandomVariableAtom latentAtom = db.getAtom(latentVariable, pic, typeID)

			latentAtom.setValue(rand.nextDouble());

			latentAtom.commitToDB()
			j++;
		}
	}
}


