package edu.umd.cs.psl.experiments.wikipedia;

import org.apache.commons.lang.builder.ToStringBuilder;
import org.apache.commons.lang.builder.ToStringStyle;

public abstract class WikiLoadConfig {

	public enum SplitType { Random, Snowball };
	
	public String categoryFile="category.txt";
	public String categoryTable="category";
	public String categoryBelongFile="newCategoryBelonging.txt";
	//public String categoryBelongFile="categoryBelonging.txt";
	public String categoryBelongTable="categorybelonging";
	public String documentsFile="pruned-document.txt";
	public String documentsTable="document";
	public String editsFile="twoYearEditEventCounts.txt";
	//public String editsFile="ordinalTimeCleanedUpEditEvent.txt";
	public String editsTable="editevent";
	public String talksFile="twoYearTopicTalkEventCounts.txt";
	//public String talksFile="ordinalTimeCleanedUpTopicTalkEvent.txt";
	public String talksTable="talkevent";
	public String classifyTable = "classifyCat";
	public String linksFile="WithinWithinLinks.txt";
	public String linksTable="withinwithinlink";
	public String textFile="documentSimilarity.txt";
	public String textTable="similartext";
	
	public abstract boolean hasEventCounts();
	public abstract int[] countBounds();
	public abstract double countNormalizer();
	public abstract boolean hasTimedEvents();
	public abstract int[] timeBounds();
	
	public abstract double textsimilarity(double sim);
	
	public String knownTable="known";
	public String unknownTable="unknown";
	public String infEditTable="infedit";
	public String infTalkTable="inftalk";
	
	public abstract boolean hasDocumentQuality();
	public abstract int filterByDocumentQuality();
	
//	public String dataPath = "./data/wikipedia/sparse/";
	public String dataPath="./data/wikipedia/full/";
	public abstract double percentageDocumentHoldout();
	public abstract double percentageClassifyTrain();
	public abstract boolean hasEventHoldout();
	public abstract boolean holdoutEvent(int time);
	
	public abstract SplitType splitType();
	public abstract int snowballSize();
	public abstract double exploreProbability();
	
	@Override
	public String toString() {
		return ToStringBuilder.reflectionToString(this,ToStringStyle.MULTI_LINE_STYLE);
	}

	
}
