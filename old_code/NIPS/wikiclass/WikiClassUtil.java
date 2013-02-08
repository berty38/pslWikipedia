package edu.umd.cs.psl.experiments.wikiclass;


import java.io.BufferedReader;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.Arrays;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Set;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import edu.umd.cs.psl.database.DataStore;
import edu.umd.cs.psl.ui.data.graph.Entity;
import edu.umd.cs.psl.ui.data.graph.Subgraph;
import edu.umd.cs.psl.database.RDBMS.RDBMSDataStore;
import edu.umd.cs.psl.database.loading.Inserter;

public class WikiClassUtil {

	private static final Logger log = LoggerFactory.getLogger(WikiClassUtil.class);

	public static void readRelationFile(String fileLinks, Inserter linkIns, Set<Integer> ids, boolean hasTruth) throws IOException {
		BufferedReader links = new BufferedReader(new FileReader(fileLinks));
		String line = null;
		while((line = links.readLine()) != null) {
			String[] parts = line.split("\\t+");
			assert (hasTruth && parts.length==3) || (!hasTruth && parts.length==2) : parts.length;
			
			int p1 = Integer.parseInt(parts[0].trim());
			int p2 = Integer.parseInt(parts[1].trim());
			double truth=1.0;
			if (hasTruth) truth = Double.parseDouble(parts[2].trim());
			
			if (ids.contains(p1) && ids.contains(p2)) {
				linkIns.insertValue(truth, p1,p2);			
			}
		}
		links.close();
	}
	
	public static void readCategoryFile(String fileCat, Inserter ins, Set<Integer> ids) throws IOException {
		BufferedReader links = new BufferedReader(new FileReader(fileCat));
		String line = null;
		while((line = links.readLine()) != null) {
			String[] parts = line.split("\\t+");
			assert parts.length==2 : parts.length;
			
			int doc = Integer.parseInt(parts[0].trim());
			int cat = Integer.parseInt(parts[1].trim());
			
			if (ids.contains(doc)) {
				ins.insert(doc,cat);
			}
			
		}
		links.close();
	}
	
	public static Set<Integer> readDocumentFile(String fileDoc, Inserter ins) throws IOException {
		HashSet<Integer> docs  = new HashSet<Integer>();
		
		BufferedReader file = new BufferedReader(new FileReader(fileDoc));
		String line = null;
		while((line = file.readLine()) != null) {
			String[] parts = line.split("\\t+");
			assert parts.length==1;
			
			int doc = Integer.parseInt(parts[0].trim());
			
			ins.insert(doc);
			docs.add(doc);
		}
		file.close();
		
		return docs;
	}
	
	public static void insertCategories(Integer[] categoryIDs, Inserter categoryIns) {
		for (Integer cat : categoryIDs) {
			categoryIns.insert(cat,"category"+cat);
		}
	}
	
	public static void fullGrounding(Integer[] categoryIDs, Set<Integer> ids, Inserter ins) {
		for (Integer cat : categoryIDs) {
			for (Integer doc : ids) {
				ins.insert(doc,cat);
			}
		}
	}
	
	public static void createSplits(String subdir, WikiLoadConfig config, int noSplits, double exploreProbability, double knownPercentage) throws IOException {
		WikiGraph datagraph = new WikiGraph();
		datagraph.loadEntityAttributes(config.getPath(config.documentsFile), WikiEntities.Document, new String[]{null,"quality"}, true);
		log.debug("Total number of documents: {}",datagraph.getNoEntities(WikiEntities.Document));
		datagraph.loadEntityAttributes(config.getPath(config.categoryBelongFile), WikiEntities.Document, new String[]{"category"}, false);
		log.debug("Total number of documents: {}",datagraph.getNoEntities(WikiEntities.Document));
		Set<Integer> validcats = new HashSet<Integer>(Arrays.asList(config.validCategories));
		Iterator<Entity<WikiEntities,WikiRelations>> iter = datagraph.getEntities(WikiEntities.Document);
		while (iter.hasNext()) {
			String cat = iter.next().getAttribute("category", String.class);
			if (!validcats.contains(Integer.parseInt(cat))) iter.remove();
		}
		int noDocs = datagraph.getNoEntities(WikiEntities.Document);
		log.debug("Total number of documents: {}",noDocs);
		datagraph.loadRelationship(config.getPath(config.linksFile), WikiRelations.Link, new WikiEntities[]{WikiEntities.Document,WikiEntities.Document}, new boolean[]{false,false});
		
		List<Subgraph<WikiEntities,WikiRelations>> subgraphs = datagraph.splitGraphSnowball(noSplits, WikiEntities.Document, 
																(int)Math.round(noDocs/(noSplits*1.25)), exploreProbability);
		
		//Now print out
		for (int split=1;split<=noSplits;split++) {
			Subgraph<WikiEntities,WikiRelations> sub = subgraphs.get(split-1);
			log.debug("Subgraph {} Size {}",split,sub.size());
			PrintWriter knowntrain = new PrintWriter(new FileWriter(config.getTrainPath(subdir,config.knownFile,split,noSplits)));
			PrintWriter unknowntrain = new PrintWriter(new FileWriter(config.getTrainPath(subdir,config.unknownFile,split,noSplits)));
			PrintWriter knowntest = new PrintWriter(new FileWriter(config.getTestPath(subdir,config.knownFile,split,noSplits)));
			PrintWriter unknowntest = new PrintWriter(new FileWriter(config.getTestPath(subdir,config.unknownFile,split,noSplits)));
			
			iter = datagraph.getEntities(WikiEntities.Document);
			while (iter.hasNext()) {
				Entity<WikiEntities,WikiRelations> doc = iter.next();
				boolean isknown = Math.random()<knownPercentage;
				if (sub.containsEntity(doc)) {
					if (isknown) knowntest.println(doc.getId());
					else unknowntest.println(doc.getId());
				} else {
					if (isknown) knowntrain.println(doc.getId());
					else unknowntrain.println(doc.getId());
				}
			}
			
			knowntrain.close();
			unknowntrain.close();
			knowntest.close();
			unknowntest.close();
		}
		
	}
	
	public static int[] parseInteger(String[] ints) {
		int[] res = new int[ints.length];
		for (int i=0;i<ints.length;i++) res[i]  = Integer.parseInt(ints[i]);
		return res;
	}
	
	public static double computeAccuracy(RDBMSDataStore data, int writeID, int testTruthID) {
		String var = "res";
		double numCorrect = data.querySingleStats("SELECT count(h.paper) as res FROM category h, category k WHERE " +
				" h.fold="+writeID+" and h.truth=(select max(truth) from category as f where f.paper=h.paper and f.fold="+writeID+") " +
				"and k.fold="+testTruthID+" and k.paper=h.paper and k.category=h.category and h.truth>0.2",
				var);
		double count = data.querySingleStats("SELECT COUNT(DISTINCT c.paper) as res from category c where c.fold="+testTruthID,var);
		//double[] numPredicted = queryStats("SELECT count(DISTINCT h.doc) as res FROM categorybelonging h WHERE " +
		//			" h.fold="+writeID, var);
		
		double acc = (numCorrect*1.0/count);
		return acc;
	}

	public static void main(String[] args) throws IOException {
		WikiLoadConfig wikiconfig = new WikiLoadConfig("/Users/matthias/Development/Eclipse/PSL/data/wikipedia/nips2010data/");

		int noSplits=20;
		double exploreProb=1.0;
		double knownPercentage=0.2;
		String dir = noSplits+"foldsP="+knownPercentage+"C7";
		
		de.mathnbits.io.DirectoryUtils.mkDirException(wikiconfig.getPath(dir));
		WikiClassUtil.createSplits(dir,wikiconfig, noSplits, exploreProb, knownPercentage);
	}
	
}
