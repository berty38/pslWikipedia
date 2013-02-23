package edu.umd.cs.linqs.vision

import org.slf4j.Logger
import org.slf4j.LoggerFactory
import com.google.common.collect.Iterables

import edu.umd.cs.linqs.vision.PatchStructure.Patch
import edu.umd.cs.linqs.wiki.DataOutputter
import edu.umd.cs.psl.application.inference.MPEInference
import edu.umd.cs.psl.application.learning.weight.maxlikelihood.MaxLikelihoodMPE
import edu.umd.cs.psl.application.learning.weight.maxlikelihood.MaxPseudoLikelihood
import edu.umd.cs.psl.application.learning.weight.maxlikelihood.VotedPerceptron
import edu.umd.cs.psl.application.learning.weight.maxmargin.MaxMargin
import edu.umd.cs.psl.application.learning.weight.maxmargin.MaxMargin.LossBalancingType
import edu.umd.cs.psl.application.learning.weight.maxmargin.MaxMargin.NormScalingType
import edu.umd.cs.psl.application.learning.weight.random.MetropolisRandOM
import edu.umd.cs.psl.application.learning.weight.random.FirstOrderMetropolisRandOM
import edu.umd.cs.psl.application.learning.weight.random.GroundMetropolisRandOM
import edu.umd.cs.psl.application.learning.weight.random.IncompatibilityMetropolisRandOM
import edu.umd.cs.psl.config.ConfigBundle;
import edu.umd.cs.psl.config.ConfigManager
import edu.umd.cs.psl.database.DataStore
import edu.umd.cs.psl.database.Database;
import edu.umd.cs.psl.database.Partition
import edu.umd.cs.psl.database.loading.Inserter
import edu.umd.cs.psl.database.rdbms.RDBMSDataStore
import edu.umd.cs.psl.database.rdbms.driver.H2DatabaseDriver
import edu.umd.cs.psl.database.rdbms.driver.H2DatabaseDriver.Type
import edu.umd.cs.psl.evaluation.result.FullInferenceResult
import edu.umd.cs.psl.evaluation.statistics.ContinuousPredictionComparator
import edu.umd.cs.psl.groovy.*
import edu.umd.cs.psl.model.Model;
import edu.umd.cs.psl.model.argument.Term
import edu.umd.cs.psl.model.argument.ArgumentType
import edu.umd.cs.psl.model.argument.UniqueID
import edu.umd.cs.psl.model.argument.Variable
import edu.umd.cs.psl.model.kernel.CompatibilityKernel
import edu.umd.cs.psl.model.parameters.Weight
//import edu.umd.cs.psl.application.learning.weight.random.GroundIncompatibilityMetropolisRandOM;
//import edu.umd.cs.psl.application.learning.weight.random.HardEMRandOM2


Logger log = LoggerFactory.getLogger(this.class)

/* VISION EXPERIMENT SETTINGS */
// test on left half of face (bottom if false)
testLeft = true
// train on randomly sampled pixels
trainOnRandom = true
// number of training faces
numTraining = 50
// number of testing faces
numTesting = 50
// max prediction resolution
maxGridResolution = 64


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

/*
 * SET LEARNING ALGORITHMS
 * 
 * Options:
 * "MLE" (MaxLikelihoodMPE)
 * "MPLE" (MaxPseudoLikelihood)
 * "MM" (MaxMargin)
 * "FirstOrderRandOM" (FirstOrderMetropolisRandOM)
 * "IncompatibilityRandOM" (IncompatibilityMetropolisRandOM)
 * "GroundRandOM" (GroundMetropolisRandOM)
 */
methods = ["MLE"];

/* MLE/MPLE options */
vpStepCounts = [200]
vpStepSizes = [5]


ConfigManager cm = ConfigManager.getManager()
ConfigBundle baseConfig = cm.getBundle("vision")

boolean sq = true

/*
 * DEFINES EXPERIMENT CONFIGURATIONS
 */
List<String> methodNames = new ArrayList<String>();
List<ConfigBundle> methodConfigs = new ArrayList<ConfigBundle>();
for (String method : methods) {
	if (method.equals("MLE") || method.equals("MPLE")) {
		for (int vpStepCount : vpStepCounts) {
			for (double vpStepSize : vpStepSizes) {
				ConfigBundle newBundle = cm.getBundle("vision");
				newBundle.addProperty("method", method);
				newBundle.addProperty(VotedPerceptron.NUM_STEPS_KEY, vpStepCount);
				newBundle.addProperty(VotedPerceptron.STEP_SIZE_KEY, vpStepSize);
				methodName = ((sq) ? "quad" : "linear") + "-" + method.toLowerCase() + "-" + vpStepCount + "-" + vpStepSize;
				methodNames.add(methodName);
				methodConfigs.add(newBundle);
			}
		}
	}
	else {
		ConfigBundle newBundle = cm.getBundle("vision");
		newBundle.addProperty("method", method);
		methodName = ((sq) ? "quad" : "linear") + "-" + method.toLowerCase();
		methodNames.add(methodName);
		methodConfigs.add(newBundle);
	}
}

/*
 * PRINTS EXPERIMENT CONFIGURATIONS
 */
for (int methodIndex = 0; methodIndex < methodNames.size(); methodIndex++)
	System.out.println("Config for " + methodNames.get(methodIndex) + "\n" + methodConfigs.get(methodIndex));

/*
 * INITIALIZES DATASTORE AND MODEL
 */
def defaultPath = System.getProperty("java.io.tmpdir") + "/"
String dbpath = baseConfig.getString("dbpath", defaultPath + "pslOlivetti")
DataStore data = new RDBMSDataStore(new H2DatabaseDriver(Type.Disk, dbpath, true), baseConfig)
//DataStore data = new RDBMSDataStore(new H2DatabaseDriver(Type.Memory, dbpath, true), baseConfig)

PSLModel m = new PSLModel(this, data)

/*
 * DEFINE MODEL
 */

width = 64
height = 64
branching = 2
depth = 7
def hierarchy = new PatchStructure(width, height, branching, depth, baseConfig)
hierarchy.generateHierarchy()


m.add predicate: "north", types: [ArgumentType.UniqueID, ArgumentType.UniqueID]
m.add predicate: "east", types: [ArgumentType.UniqueID, ArgumentType.UniqueID]
m.add predicate: "horizontalMirror", types: [ArgumentType.UniqueID, ArgumentType.UniqueID]
m.add predicate: "verticalMirror", types: [ArgumentType.UniqueID, ArgumentType.UniqueID]
m.add predicate: "children", types: [ArgumentType.UniqueID, ArgumentType.UniqueID]
m.add predicate: "neighbors", types: [ArgumentType.UniqueID, ArgumentType.UniqueID]
m.add predicate: "level", types: [ArgumentType.UniqueID, ArgumentType.UniqueID]
m.add predicate: "pixelBrightness", types: [ArgumentType.UniqueID, ArgumentType.UniqueID]
m.add predicate: "picture", types: [ArgumentType.UniqueID]

m.add setcomparison: "brightness", on: pixelBrightness, using: SetComparison.Average
m.createFormulaContainer("brightness", {A.children}, {P})
//brightness({A.children}, {P})
//m.add setcomparison: "neighborBrightness", on: brightness__1, using: SetComparison.Average
//m.createFormulaContainer("neighborBrightness", {A.neighbors}, {P})

//m.add rule : ~(pixelBrightness(A,P)), weight: 0.001, squared: sq

double initialWeight = 1.0

for (Patch p : hierarchy.getPatches().values()) {
	UniqueID patch = data.getUniqueID(p.uniqueID())
	UniqueID L = data.getUniqueID(p.getLevel())
	Variable pic = new Variable("pictureVar")
	Term [] args = new Term[2]
	args[0] = patch
	args[1] = pic

	// set two-sided prior
	//	m.add rule: (picture(pic) & level(patch,L)) >> brightness__1(patch,pic), weight: initialWeight, squared: sq
	//	m.add rule: (picture(pic) & level(patch,L)) >> ~brightness__1(patch,pic), weight: initialWeight, squared: sq

	if (p.getLevel() <= maxGridResolution) {
		/** NEIGHBOR AGREEMENT **/
		// north neighbor
		if (p.hasNorth()) {
			m.add rule: (picture(pic) & level(patch,L) & north(patch,N) & brightness__1(N,pic)) >> brightness__1(patch,pic), weight: initialWeight, squared: sq
			m.add rule: (picture(pic) & level(patch,L) & north(patch,N) & brightness__1(N,pic)) >> ~brightness__1(patch,pic), weight: initialWeight, squared: sq
			m.add rule: (picture(pic) & level(patch,L) & north(patch,N) & ~brightness__1(N,pic)) >> brightness__1(patch,pic), weight: initialWeight, squared: sq
			m.add rule: (picture(pic) & level(patch,L) & north(patch,N) & ~brightness__1(N,pic)) >> ~brightness__1(patch,pic), weight: initialWeight, squared: sq
		}

		// south neighbor
		if (p.hasSouth()) {
			m.add rule: (picture(pic) & level(patch,L) & north(N, patch) & brightness__1(N,pic)) >> brightness__1(patch,pic), weight: initialWeight, squared: sq
			m.add rule: (picture(pic) & level(patch,L) & north(N, patch) & brightness__1(N,pic)) >> ~brightness__1(patch,pic), weight: initialWeight, squared: sq
			m.add rule: (picture(pic) & level(patch,L) & north(N, patch) & ~brightness__1(N,pic)) >> brightness__1(patch,pic), weight: initialWeight, squared: sq
			m.add rule: (picture(pic) & level(patch,L) & north(N, patch) & ~brightness__1(N,pic)) >> ~brightness__1(patch,pic), weight: initialWeight, squared: sq
		}

		// east neighbor
		if (p.hasEast()) {
			m.add rule: (picture(pic) & level(patch,L) & east(patch,N) & brightness__1(N,pic)) >> brightness__1(patch,pic), weight: initialWeight, squared: sq
			m.add rule: (picture(pic) & level(patch,L) & east(patch,N) & brightness__1(N,pic)) >> ~brightness__1(patch,pic), weight: initialWeight, squared: sq
			m.add rule: (picture(pic) & level(patch,L) & east(patch,N) & ~brightness__1(N,pic)) >> brightness__1(patch,pic), weight: initialWeight, squared: sq
			m.add rule: (picture(pic) & level(patch,L) & east(patch,N) & ~brightness__1(N,pic)) >> ~brightness__1(patch,pic), weight: initialWeight, squared: sq
		}

		// west neighbor
		if (p.hasWest()) {
			m.add rule: (picture(pic) & level(patch,L) & east(N, patch) & brightness__1(N,pic)) >> brightness__1(patch,pic), weight: initialWeight, squared: sq
			m.add rule: (picture(pic) & level(patch,L) & east(N, patch) & brightness__1(N,pic)) >> ~brightness__1(patch,pic), weight: initialWeight, squared: sq
			m.add rule: (picture(pic) & level(patch,L) & east(N, patch) & ~brightness__1(N,pic)) >> brightness__1(patch,pic), weight: initialWeight, squared: sq
			m.add rule: (picture(pic) & level(patch,L) & east(N, patch) & ~brightness__1(N,pic)) >> ~brightness__1(patch,pic), weight: initialWeight, squared: sq
		}

		m.add rule: (horizontalMirror(patch,B) & brightness__1(patch,pic)) >> brightness__1(B,pic), weight: initialWeight, squared: sq
		m.add rule: (horizontalMirror(patch,B) & ~brightness__1(patch,pic) & picture(pic)) >> ~brightness__1(B,pic), weight: initialWeight, squared: sq
		m.add rule: (verticalMirror(patch,B) & brightness__1(patch,pic)) >> brightness__1(B,pic), weight: initialWeight, squared: sq
		m.add rule: (verticalMirror(patch,B) & ~brightness__1(patch,pic) & picture(pic)) >> ~brightness__1(B,pic), weight: initialWeight, squared: sq
	}
}


// save all initial weights
Map<CompatibilityKernel,Weight> weights = new HashMap<CompatibilityKernel, Weight>()
for (CompatibilityKernel k : Iterables.filter(m.getKernels(), CompatibilityKernel.class))
	weights.put(k, k.getWeight());

log.info("Model has {} weighted kernels", weights.size());

Partition trainObs =  new Partition(0)
Partition testObs = new Partition(1)
Partition trainLabel = new Partition(2)
Partition testLabel = new Partition(3)
Partition trainWrite = new Partition(4)
Partition testWrite = new Partition(5)

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
		c++
	}
}

for (Partition part : [trainObs, testObs]) {
	def readDB = data.getDatabase(part)
	ImagePatchUtils.insertFromPatchMap(north, readDB, hierarchy.getNorth())
	ImagePatchUtils.insertFromPatchMap(east, readDB, hierarchy.getEast())
	ImagePatchUtils.insertFromPatchMap(horizontalMirror, readDB, hierarchy.getMirrorHorizontal())
	ImagePatchUtils.insertFromPatchMap(verticalMirror, readDB, hierarchy.getMirrorVertical())
	//		ImagePatchUtils.insertNeighbors(neighbors, readDB, hierarchy)
	ImagePatchUtils.insertPatchLevels(readDB, hierarchy, level)
	ImagePatchUtils.insertPixelPatchChildren(children, readDB, hierarchy)
	readDB.close()
}

ArrayList<double []> images = ImagePatchUtils.loadImages(dataDir + "/olivetti01.txt", width, height)
// create list of train images and test images
ArrayList<double []> trainImages = new ArrayList<double[]>()
ArrayList<double []> testImages = new ArrayList<double[]>()

for (int i = 0; i < images.size(); i++) {
	if (i < numTraining)
		trainImages.add(images.get(i))
	else if (i >= images.size() - numTesting)
		testImages.add(images.get(i))
}

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
for (int i = 0; i < trainImages.size(); i++) {

	c = 0
	for (int x = 0; x < width; x++) {
		for (int y = 0; y < height; y++) {
			if (trainOnRandom) {
				trainMask[c] = rand.nextBoolean()
			} else
				trainMask[c] = mask[c]
			negTrainMask[c] = !trainMask[c]
			c++
		}
	}

	UniqueID id = data.getUniqueID(i)
	ImagePatchUtils.setPixels(pixelBrightness, id, trainReadDB, hierarchy, width, height, trainImages.get(i), trainMask)
	ImagePatchUtils.setPixels(pixelBrightness, id, trainLabelDB, hierarchy, width, height, trainImages.get(i), negTrainMask)

	ImagePatchUtils.populateAllPatches(brightness__1, id, trainWriteDB, hierarchy)
	ImagePatchUtils.populateAllPatches(brightness__1, id, trainLabelDB, hierarchy)
	//	ImagePatchUtils.populateAllPatches(neighborBrightness__2, id, trainWriteDB, hierarchy)
	//	ImagePatchUtils.populateAllPatches(neighborBrightness__2, id, trainLabelDB, hierarchy)

	ImagePatchUtils.computePatchBrightness(brightness__1, pixelBrightness, trainLabelDB, id, hierarchy, trainImages.get(i))
	//	ImagePatchUtils.computeNeighborBrightness(neighborBrightness__2, brightness__1, neighbors, trainLabelDB, id, hierarchy)
}
trainWriteDB.close()
trainReadDB.close()
trainLabelDB.close()

def testReadDB = data.getDatabase(testObs)
def testLabelDB = data.getDatabase(testLabel)
def testWriteDB = data.getDatabase(testWrite)
for (int i = 0; i < testImages.size(); i++) {
	def id = data.getUniqueID(i)
	ImagePatchUtils.setPixels(pixelBrightness, id, testReadDB, hierarchy, width, height, testImages.get(i), mask)
	ImagePatchUtils.setPixels(pixelBrightness, id, testLabelDB, hierarchy, width, height, testImages.get(i), negMask)

	ImagePatchUtils.populateAllPatches(brightness__1, id, testWriteDB, hierarchy)
	ImagePatchUtils.populateAllPatches(brightness__1, id, testLabelDB, hierarchy)
	//	ImagePatchUtils.populateAllPatches(neighborBrightness__2, id, testWriteDB, hierarchy)
	//	ImagePatchUtils.populateAllPatches(neighborBrightness__2, id, testLabelDB, hierarchy)

	ImagePatchUtils.computePatchBrightness(brightness__1, pixelBrightness, testLabelDB, id, hierarchy, testImages.get(i))
	//	ImagePatchUtils.computeNeighborBrightness(neighborBrightness__2, brightness__1, neighbors, testLabelDB, id, hierarchy)
}
testWriteDB.close()
testReadDB.close()
testLabelDB.close()

/** start experiments **/
def scores = new ArrayList<Double>()

for (int methodIndex = 0; methodIndex < methodNames.size(); methodIndex++) {

	/** open databases **/
	def labelClose = [pixelBrightness, brightness__1] as Set
	def toClose = [north, east, horizontalMirror, verticalMirror, children, level, picture] as Set
	def trainDB = data.getDatabase(trainWrite, toClose, trainObs)
	def testDB = data.getDatabase(testWrite, toClose, testObs)
	def labelDB = data.getDatabase(trainLabel, labelClose)
	def groundTruthDB = data.getDatabase(testLabel, labelClose)

	for (CompatibilityKernel k : Iterables.filter(m.getKernels(), CompatibilityKernel.class))
		k.setWeight(weights.get(k))

	/*
	 * Weight learning
	 */
	/** populate pixelBrightness **/
	for (int i = 0; i < trainImages.size(); i++) {
		UniqueID id = data.getUniqueID(i)
		ImagePatchUtils.populatePixels(width, height, pixelBrightness, trainDB, id)
	}
	learn(m, trainDB, labelDB, methodConfigs.get(methodIndex), log)
	System.out.println("Learned model " + methodNames.get(methodIndex) + "\n" + m.toString())

	trainDB.close()
	labelDB.close()

	/*
	 * Inference on test set
	 */
	for (int i = 0; i < testImages.size(); i++) {
		UniqueID id = data.getUniqueID(i)
		ImagePatchUtils.populatePixels(width, height, pixelBrightness, testDB, id)
	}
	MPEInference mpe = new MPEInference(m, testDB, baseConfig)
	FullInferenceResult result = mpe.mpeInference()
	System.out.println("Objective: " + result.getTotalWeightedIncompatibility())
	def expSetup = (testLeft? "left" : "bottom") + "-" + (trainOnRandom? "same" : "rand")
	DataOutputter.outputPredicate("output/vision/olivetti-" + expSetup + "-" + methodNames.get(methodIndex) + ".txt" , testDB, pixelBrightness, ",", true, "index,image")

	testDB.close()

	/*
	 * Evaluation
	 */
	testDB = data.getDatabase(testWrite)
	def comparator = new ContinuousPredictionComparator(testDB)
	comparator.setBaseline(groundTruthDB)

	def metric = ContinuousPredictionComparator.Metric.MSE
	comparator.setMetric(metric)
	score = comparator.compare(pixelBrightness) * (255 * 255) // scale to full 256 grayscale range
	scores.add(methodIndex, score);

	methodName = methodNames.get(methodIndex)
	System.out.println("Method: " + methodName + ", mean squared error: " + scores.get(methodIndex))

	// close all databases
	groundTruthDB.close()
	testDB.close()
}

def expSetup = (testLeft? "left" : "bottom") + "-" + (trainOnRandom? "same" : "rand")

for (int methodIndex = 0; methodIndex < methodNames.size(); methodIndex++) {
	methodName = methodNames.get(methodIndex)
	System.out.println(expSetup + ", method: " + methodName + ", mean squared error: " + scores.get(methodIndex))
}

private void learn(Model m, Database db, Database labelsDB, ConfigBundle config, Logger log) {
	switch(config.getString("method", "")) {
		case "MLE":
			MaxLikelihoodMPE mle = new MaxLikelihoodMPE(m, db, labelsDB, config)
			mle.learn()
			break
		case "MPLE":
			MaxPseudoLikelihood mple = new MaxPseudoLikelihood(m, db, labelsDB, config)
			mple.learn()
			break
		case "MM":
			MaxMargin mm = new MaxMargin(m, db, labelsDB, config)
			mm.learn()
			break
		case "HEMRandOM":
		//			HardEMRandOM2 hardRandOM = new HardEMRandOM2(m, db, labelsDB, config)
		//			hardRandOM.setSlackPenalty(10000)
		//			hardRandOM.learn()
			break
		case "FirstOrderRandOM":
			FirstOrderMetropolisRandOM randOM = new FirstOrderMetropolisRandOM(m, db, labelsDB, config)
			randOM.learn()
			break
		case "IncompatibilityRandOM":
		//			GroundIncompatibilityMetropolisRandOM randOM = new GroundIncompatibilityMetropolisRandOM(m, db, labelsDB, config);
			IncompatibilityMetropolisRandOM randOM = new IncompatibilityMetropolisRandOM(m, db, labelsDB, config);
			randOM.learn();
			break
		case "GroundRandOM":
			GroundMetropolisRandOM randOM = new GroundMetropolisRandOM(m, db, labelsDB, config)
			randOM.learn()
			break
		case "None":
			break;
		default:
			throw new IllegalArgumentException("Unrecognized method.");
	}
}
