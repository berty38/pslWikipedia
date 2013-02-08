package edu.umd.cs.psl.experiments.wikipedia;

import org.apache.commons.lang.builder.ToStringBuilder;
import org.apache.commons.lang.builder.ToStringStyle;

public class WikiLoadEditConfig extends WikiLoadConfig {

	public boolean hasEventCounts() {
		return true;
	}
	public final int[] countBounds() {
		return new int[]{5,30000000};
	}
	public final double countNormalizer() {
		return 10.0;
	}
	public boolean hasTimedEvents() {
		return false;
	}
	public int[] timeBounds() {
		//return new int[]{733300,1000000};
		return new int[]{733400,1000000};
	}
	
	public boolean hasDocumentQuality() {
		return  true;
	}
	public int filterByDocumentQuality() {
		return 1;
	}
	
//	public String dataPath = "./data/wikipedia/sparse/";
	public String dataPath="./data/wikipedia/full/";
	
	public double percentageDocumentHoldout() {
		return 0.00;
	}
	
	public double percentageClassifyTrain() {
		return 0.0;
	}
	
	public boolean hasEventHoldout() {
		return true;
	}
	
	public double textsimilarity(double sim) {
		if (sim>=0.4) return 1.0;
		else return 0.0;
	}
	
	public boolean holdoutEvent(int time) {
		//return time>733600;
		return Math.random()<0.2;
	}
	
	public SplitType splitType() {
		return SplitType.Random;
	}
	
	public int snowballSize() {
		return 600;
	}
	public double exploreProbability() {
		return 0.3;
	}
	
	@Override
	public String toString() {
		return ToStringBuilder.reflectionToString(this,ToStringStyle.MULTI_LINE_STYLE);
	}
	
}
