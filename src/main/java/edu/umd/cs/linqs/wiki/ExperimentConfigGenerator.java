package edu.umd.cs.linqs.wiki;

import java.util.ArrayList;
import java.util.List;

import edu.umd.cs.psl.application.learning.weight.maxmargin.MaxMargin.LossBalancingType;
import edu.umd.cs.psl.application.learning.weight.maxmargin.MaxMargin.NormScalingType;
import edu.umd.cs.psl.config.ConfigBundle;

/**
 * Generates a set of {@link ConfigBundle ConfigBundles}.
 * 
 * @author Stephen Bach <bach@cs.umd.edu>
 */
public class ExperimentConfigGenerator {
	
	public enum modelType {LINEAR, QUAD, BOOLEAN};
	
	public enum inferenceType {NONE, MPE, LAZY_MPE};
	
	public enum reasonerType {ADMM, BOOLEAN_MAX, BOOLEAN_MC};
	
	public ExperimentConfigGenerator(String baseConfigName, modelType mType) {
		this.baseConfigName = baseConfigName;
		this.mType = mType;
	}

	/* General options */
	protected final String baseConfigName;
	protected final modelType mType;
	
	protected List<String> learningMethods = new ArrayList<String>();
	public void setLearningMethods(List<String> learningMethods) { this.learningMethods = learningMethods; }
	
	protected inferenceType iType = inferenceType.NONE;
	public void setInferenceType(inferenceType iType) { this.iType = iType; }
	
	/* VotedPerceptron options */
	protected List<Integer> votedPerceptronStepCounts = new ArrayList<Integer>();
	public void setVotedPerceptronStepCounts(List<Integer> vpStepCounts) { votedPerceptronStepCounts = vpStepCounts; }
	
	protected List<Double> votedPerceptronStepSizes = new ArrayList<Double>();
	public void setVotedPerceptronStepSizes(List<Double> vpStepSizes) { votedPerceptronStepSizes = vpStepSizes; }
	
	/* MaxMargin options */
	protected List<Double> maxMarginSlackPenalties = new ArrayList<Double>();
	public void setMaxMarginSlackPenalties(List<Double> mmSlackPenalties) { maxMarginSlackPenalties = mmSlackPenalties; }
	
	protected List<LossBalancingType> maxMarginLossBalancingTypes = new ArrayList<LossBalancingType>();
	protected List<NormScalingType> maxMarginNormScalingTypes = new ArrayList<NormScalingType>();
	
	/* BooleanMaxWalkSat options */
	protected List<Integer> maxFlips = new ArrayList<Integer>();
	public void setMaxFlips(List<Integer> maxFlips) { this.maxFlips = maxFlips; }
	
	protected List<Double> noiseValues = new ArrayList<Double>();
	public void setNoiseValues(List<Double> noiseValues) { this.noiseValues = noiseValues; }
}
