package edu.umd.cs.psl.experiments.wikiclass;

//import java.io.File;

import java.io.File;

import org.apache.commons.lang.builder.ToStringBuilder;
import org.apache.commons.lang.builder.ToStringStyle;

public class WikiLoadConfig {

	
	public String categoryBelongFile="newCategoryBelonging.txt";
	public String categoryBelongTable="category";

	public String documentsFile="pruned-documentFilteredBy1.txt";
	public String documentsTable="document";

//	public String classifyTable = "classifyCat";

	public String linksFile="WithinWithinLinks.txt";
	public String linksTable="withinwithinlink";

	public String textFile="documentSimilarity.txt";
	public String textTable="similartext";
		
	public String knownTable="known";
	public String knownFile = "known.txt";
	public String unknownTable="unknown";
	public String unknownFile = "unknown.txt";
	
	//public Integer[] validCategories = {1,2,3,4,6,7,8,9,11,13,18,19};
	public Integer[] validCategories = {1,2,3,6,9,11,18};
	
	public final String dataPath;
	
	//="/Users/matthias/Development/Eclipse/PSL/data/wikipedia/nips2010data/";

	public WikiLoadConfig(String basedir) {
		dataPath = basedir;
	}
	
	public String getPath(String file) {
		return dataPath + file;
	}
	
	public String getTestPath(String subdir, String file, int fold, int totalFolds) {
		return getPath(subdir, "test"+fold+"of"+totalFolds+file);
	}
	
	public String getTrainPath(String subdir, String file, int fold, int totalFolds) {
		return getPath(subdir, "train"+fold+"of"+totalFolds+file);
	}
	
	public String getPath(String subdir, String file) {
		return dataPath + subdir + File.separator + file;
	}
	
	@Override
	public String toString() {
		return ToStringBuilder.reflectionToString(this,ToStringStyle.MULTI_LINE_STYLE);
	}

	
}
