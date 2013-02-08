package edu.umd.cs.psl.experiments.wikipedia;

import org.apache.commons.lang.builder.ToStringBuilder;
import org.apache.commons.lang.builder.ToStringStyle;

public class WikiLoadCategoryConfig extends WikiLoadConfig {

	public boolean hasEventCounts() {
		return true;
	}
	
	private double textsimilarityThres = 0.4;
	
	public double textsimilarity(double sim) {
		if (sim<textsimilarityThres) return 0.0;
		else return sim;
	}
	
	private int[] countBounds = {5,30000000};
	
	public final int[] countBounds() {
		return countBounds;
	}
	
	private double countNormalizer = 10.0;
	
	public final double countNormalizer() {
		return countNormalizer;
	}
	
	private boolean hasTimedEvents = false;
	public boolean hasTimedEvents() {
		return hasTimedEvents;
	}
	
	private int[] timeBounds = {0,1000000};
	public int[] timeBounds() {
		return timeBounds;
	}
	
	private boolean hasDocumentQuality = true;
	public boolean hasDocumentQuality() {
		return  hasDocumentQuality;
	}
	
	private int filterByDocumentQuality = 1;
	public int filterByDocumentQuality() {
		return filterByDocumentQuality;
	}
	
//	public String dataPath = "./data/wikipedia/sparse/";
	public String dataPath="./data/wikipedia/full/";
	
	public double percentageDocumentHoldout = 0.75;
	public double percentageDocumentHoldout() {
		return percentageDocumentHoldout;
	}
	
	public double percentageClassifyTrain = 0.2;
	public double percentageClassifyTrain() {
		return percentageClassifyTrain;
	}
	
	private boolean hasEventHoldout = false;
	public boolean hasEventHoldout() {
		return hasEventHoldout;
	}
	
	public boolean holdoutEvent(int time) {
		throw new UnsupportedOperationException();
	}
	
	public SplitType splitType = SplitType.Random;
	public SplitType splitType() {
		return splitType;
	}
	
	private int snowballSize = 600;
	public int snowballSize() {
		return snowballSize;
	}
	private double exploreProbability = 0.3;
	public double exploreProbability() {
		return exploreProbability;
	}
	
	@Override
	public String toString() {
		return ToStringBuilder.reflectionToString(this,ToStringStyle.MULTI_LINE_STYLE);
	}
	
}
