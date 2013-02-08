package edu.umd.cs.psl.experiments.wikipedia;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.*;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.google.common.collect.Iterables;

import de.mathnbits.io.BasicUserInteraction;

import edu.umd.cs.psl.database.loading.DataLoader;
import edu.umd.cs.psl.database.loading.OpenInserter;
import edu.umd.cs.psl.experiments.ExperimentsUtil;
import edu.umd.cs.psl.experiments.collectiveclass.CollectiveClassPaper;
import edu.umd.cs.psl.ui.data.file.util.DelimitedObjectConstructor;
import edu.umd.cs.psl.ui.data.file.util.LoadDelimitedData;
import edu.umd.cs.psl.ui.data.graph.DegreeFunction;
import edu.umd.cs.psl.ui.data.graph.Entity;
import edu.umd.cs.psl.ui.data.graph.Relation;
import edu.umd.cs.psl.ui.data.graph.Subgraph;
import edu.umd.cs.psl.ui.experiment.folds.SingleFoldLoader;
import edu.umd.cs.psl.ui.functions.textsimilarity.CosineSimilarity;
import edu.umd.cs.psl.ui.functions.textsimilarity.WordClassifier;
import edu.umd.cs.psl.ui.functions.textsimilarity.CosineSimilarity.WordVector;
import edu.umd.cs.psl.ui.functions.textsimilarity.WordClassifier.ParseType;
import static edu.umd.cs.psl.experiments.wikipedia.WikiEntities.*;
import static edu.umd.cs.psl.experiments.wikipedia.WikiRelations.*;

public class WikipediaSingleFoldLoader implements SingleFoldLoader {

	private static final Logger log = LoggerFactory.getLogger(WikipediaSingleFoldLoader.class);

	
	private final static int traindataid = 1;
	private final static int traintruthid = 2;
	private final static int testdataid = 4;
	private final static int testtruthid = 5;
	

	
	private final int numSplits = 2;
	private final int[] editDegreeBounds = {2,10};
	private final double textsimthreshold = 0.3;

	

	private final WikiGraph datagraph;
	private final WikiLoadConfig config;
	
	private final List<Subgraph<WikiEntities,WikiRelations>> splits;
	


	
	public WikipediaSingleFoldLoader(WikiLoadConfig conf) {
		config = conf;
		datagraph = new WikiGraph();

		loadData();
		
		//Split documents
		switch(config.splitType()) {
		case Random: 
			splits = datagraph.splitGraphRandom(numSplits,Document);
			break;
		case Snowball:
			splits = datagraph.splitGraphSnowball(numSplits,Document,config.snowballSize(), config.exploreProbability());
			break;
		default: throw new IllegalArgumentException("Unrecognized split type: " + config.splitType());
		}
		

	}
	
	private void loadSplit(final Subgraph<WikiEntities,WikiRelations> split,final int dataID, final int truthID, WordClassifier classify, DataLoader loader) {
		OpenInserter categoryIns = loader.getOpenInserter(config.categoryTable);
		OpenInserter knownIns = loader.getOpenInserter(config.knownTable);
		OpenInserter unknownIns = config.percentageDocumentHoldout()>0?loader.getOpenInserter(config.unknownTable):null;
		OpenInserter catBelongIns =  loader.getOpenInserter(config.categoryBelongTable);
		OpenInserter documentIns =  loader.getOpenInserter(config.documentsTable);
		OpenInserter editIns =  loader.getOpenInserter(config.editsTable);
		OpenInserter talkIns =  loader.getOpenInserter(config.talksTable);
		OpenInserter linkIns =  loader.getOpenInserter(config.linksTable);
		OpenInserter textIns =  loader.getOpenInserter(config.textTable);
		OpenInserter classifyIns =  loader.getOpenInserter(config.classifyTable);
		OpenInserter ieditIns = config.hasEventHoldout()?loader.getOpenInserter(config.infEditTable):null;
		OpenInserter italkIns = config.hasEventHoldout()?loader.getOpenInserter(config.infTalkTable):null;
		
		int unknownDocs = 0;
		int linksLoaded = 0;
		int textsimloaded = 0;
		int catLoaded = 0;
		int docloaded = 0;
		
		//Insert categories
		for (Entity<WikiEntities,WikiRelations> cat : split.getEntities(Category)) {
			assert cat.hasType(Category);
			categoryIns.insert(dataID, cat.getId(), cat.getAttribute("name", String.class));
			catLoaded++;
		}
		log.debug("# categories loaded: {}", catLoaded);
		
		for (Entity<WikiEntities,WikiRelations> doc : split.getEntities(Document)) {
			assert doc.hasType(Document);
			docloaded++;
			int catID = Iterables.getOnlyElement(doc.getRelations(HasCategory)).get(1).getId();
			if (Math.random() < config.percentageDocumentHoldout()) {
				//Unknown
				unknownDocs++;
				unknownIns.insert(dataID, doc.getId());
				catBelongIns.insert(truthID, doc.getId(), catID);
				
			} else {
				knownIns.insert(dataID, doc.getId());
				catBelongIns.insert(dataID, doc.getId(), catID);
			}
			
			if (classify!=null) {
				double[] dist = classify.similarity(doc.getAttribute("words", String.class));
				for (int i=0;i<dist.length;i++) {
					if (dist[i]>0.0) classifyIns.insertTruth(dataID, dist[i], doc.getId(),Integer.parseInt(classify.getLabel(i)));
				}
			}
			//documentIns.insert(dataID, doc.getId(),doc.getAttribute("words", String.class),0);
		}
		log.debug("Loaded documents #: {} | unknown: {}",docloaded,unknownDocs);
		
		for (Relation<WikiEntities,WikiRelations> link : split.getRelations(Link)) {
			assert link.hasType(Link);
			if (link.get(0).equals(link.get(1))) continue; //Remove self loops
			linksLoaded++;
			linkIns.insert(dataID, link.get(0).getId(), link.get(1).getId());
		}
		for (Relation<WikiEntities,WikiRelations> textsim : split.getRelations(SimilarText)) {
			assert textsim.hasType(SimilarText);
			double similarity = Double.parseDouble(textsim.getAttribute("similarity", String.class));
			if (config.textsimilarity(similarity)<=0.0) continue;
			textsimloaded++;
			textIns.insertTruth(dataID, config.textsimilarity(similarity), textsim.get(0).getId(), textsim.get(1).getId());
			textIns.insertTruth(dataID, config.textsimilarity(similarity), textsim.get(1).getId(), textsim.get(0).getId());
		}
		log.debug("# Links Loaded: {} | # text sim loaded {}",linksLoaded,textsimloaded);
		
		//Load edits
		int loadedUser = 0;
		int[] loadedEvents = new int[2];
		int[] knownEvents = new int[2];
		for (Entity<WikiEntities,WikiRelations> user : split.getEntities(User)) {
			assert user.hasType(User);
			boolean loaduser = false;
			
			WikiRelations[] docevents = {Edit, Talk};
			OpenInserter[] eventIns = {editIns, talkIns};
			OpenInserter[] ieventIns = {ieditIns, italkIns};
			for (int i=0;i<2;i++) {
				int degree = user.getRelations(docevents[i]).size();
				if (degree>=editDegreeBounds[0] && degree<=editDegreeBounds[1]) {
					//Add all
					for (Relation<WikiEntities,WikiRelations> event : user.getRelations(docevents[i],split)) {
						loaduser = true;
						assert event.hasType(docevents[i]);
						loadedEvents[i]++;
						double truth = 1.0; 
						//double truth = Math.min(1.0, Integer.parseInt(event.getAttribute("count", String.class))/config.countNormalizer());
						if (config.hasEventHoldout()) {
							int time = config.hasTimedEvents()?Integer.parseInt(event.getAttribute("time", String.class)):-1;
							if (config.holdoutEvent(time)) {
								ieventIns[i].insertTruth(truthID, truth, event.get(0).getId(),event.get(1).getId());
							} else {
								knownEvents[i]++;
								eventIns[i].insertTruth(dataID, truth, event.get(0).getId(),event.get(1).getId());
								//keventIns[i].insertTruth(dataID, truth, event.get(0).getId(),event.get(1).getId());
							}
						} else {
							eventIns[i].insertTruth(dataID, truth, event.get(0).getId(),event.get(1).getId());
						}
					}
				}
			}
			if (loaduser) loadedUser++;
		}
		log.debug("# Users: {}",loadedUser);
		log.debug("# Talks: {} | # Edits: {}",loadedEvents[1],loadedEvents[0]);
		log.debug("# Known Talks: {} | # Known Edits: {}",knownEvents[1],knownEvents[0]);
	}
	
	private WordClassifier trainDocumentClassifier(final Subgraph<WikiEntities,WikiRelations> split) {
		WordClassifier.WordLoader wloader = new WordClassifier.WordLoader();
		for (Entity<WikiEntities,WikiRelations> doc : split.getEntities(Document)) {
			assert doc.hasType(Document);
			int catID = Iterables.getOnlyElement(doc.getRelations(HasCategory)).get(1).getId();
			if (Math.random() < config.percentageClassifyTrain()) {
				wloader.addWordMap(doc.getAttribute("words", String.class),catID+"");
			}
		}
		WordClassifier classify = new WordClassifier(ParseType.WordMap);
		classify.train(wloader);
		return classify;
	}
	
	@Override
	public int[] getTestDataGroundTruthIDs() {
		return new int[]{testdataid,testtruthid};
	}

	@Override
	public int[] getTestDataIDs() {
		return new int[]{testdataid};
	}

	@Override
	public int[] getTrainDataGroundTruthIDs() {
		return new int[]{traindataid, traintruthid};
	}

	@Override
	public int[] getTrainDataIDs() {
		return new int[]{traindataid};
	}

	@Override
	public void loadTest(DataLoader loader) {
		WordClassifier classify = null;
		if (config.percentageClassifyTrain()>0.0) classify=trainDocumentClassifier(splits.get(0));
		loadSplit(splits.get(1),testdataid,testtruthid,classify,loader);

	}

	@Override
	public void loadTrain(DataLoader loader) {
		WordClassifier classify = trainDocumentClassifier(splits.get(1));
		loadSplit(splits.get(0),traindataid,traintruthid,classify,loader);
	}

	public String getPath(String file) {
		return config.dataPath + File.separator + file;
	}
	
	private void loadData() {
		//Load Categories
		datagraph.loadEntityAttributes(getPath(config.categoryFile), Category, new String[]{"name"}, true);
		
		log.debug("Total number of categories: {}",datagraph.getNoEntities(Category));
		//Load Documents
		if (config.hasDocumentQuality()) {
			datagraph.loadEntityAttributes(getPath(config.documentsFile), Document, new String[]{"words",null}, new DelimitedObjectConstructor.Filter() {
				
				@Override
				public boolean include(String[] data) {
					int qual = Integer.parseInt(data[2]);
					if (qual == config.filterByDocumentQuality()) {
						return true;
					} else return false;
				}
			}, true);
		} else {
			datagraph.loadEntityAttributes(getPath(config.documentsFile), Document, new String[]{null}, true);
		}
		
		log.debug("Total number of documents: {}",datagraph.getNoEntities(Document));
		
		//Load Category assignments
		datagraph.loadRelationship(getPath(config.categoryBelongFile), HasCategory, new WikiEntities[]{Document,Category}, new boolean[]{false,false});
		//Verify that all documents have a category
		Iterator<Entity<WikiEntities,WikiRelations>> iter = datagraph.getEntities(Document);
		while (iter.hasNext()) {
			Entity<WikiEntities,WikiRelations> e = iter.next();
			assert e.getRelations(HasCategory).size()==1 : "Actual size: " + e.getRelations(HasCategory).size();
		}
		
		//Load text similarity
		datagraph.loadRelationship(getPath(config.textFile), new String[]{"similarity"}, SimilarText, new WikiEntities[]{Document,Document}, new boolean[]{false,false});
		
		
		//Prepare Events
		ArrayList<String> eventAtt = new ArrayList<String>();
		if (config.hasEventCounts()) {
			eventAtt.add("count");
		}
		if (config.hasTimedEvents()) {
			eventAtt.add("time");
		}
		String[] attNames = eventAtt.toArray(new String[eventAtt.size()]);
		DelimitedObjectConstructor.Filter eventFilter = new DelimitedObjectConstructor.Filter(){

			@Override
			public boolean include(String[] data) {
				int index = 2;
				if (config.hasEventCounts()) {
					int count = Integer.parseInt(data[index]);
					index++;
					if (count<config.countBounds()[0] || count>config.countBounds()[1])
						return false;
				}
				if (config.hasTimedEvents()) {
					int time = Integer.parseInt(data[index]);
					index++;
					if (time<config.timeBounds()[0] || time>config.timeBounds()[1])
						return false;
				}
				return true;
			}
			
		};

		
		
		//Load Edits
		datagraph.loadRelationship(getPath(config.editsFile), attNames, Edit, new WikiEntities[]{Document,User}, eventFilter, new boolean[]{false,true});
		//Load Talks
		datagraph.loadRelationship(getPath(config.talksFile), attNames, Talk, new WikiEntities[]{Document,User}, eventFilter, new boolean[]{false,true});
		//Load Links
		datagraph.loadRelationship(getPath(config.linksFile), Link, new WikiEntities[]{Document,Document}, new boolean[]{false,false});
	}
	

	
	//############# STATIC FUNCTIONS
	
	public static void computeTextSimilarity(final WikiLoadConfig config, String inputfile, String outputfile, 
											double simThreshold, int wordThreshold) {
		final Map<Integer, WordVector> documents = new HashMap<Integer, WordVector>();
		
		DelimitedObjectConstructor<Object> docLoader = new DelimitedObjectConstructor<Object>() {
						
			@Override
			public Object create(String[] data) {
				Integer docid = Integer.parseInt(data[0]);
				assert !documents.containsKey(docid) : "Duplicate document: " + docid;
				WordVector vec = CosineSimilarity.getVector(data[1]);
				documents.put(docid, vec);
				return null;
			}
			@Override
			public int length() {return config.hasDocumentQuality()?3:2;}
			
		};
		LoadDelimitedData.loadTabData(config.dataPath + inputfile, docLoader);
		log.debug("Loaded {} documents", documents.size());
		int numComputed = 0;
		double totalSim = 0.0;
	    try{
    	    BufferedWriter out = new BufferedWriter(new FileWriter(config.dataPath + outputfile));
			for (Integer doc1 : documents.keySet()) {
				for (Integer doc2 : documents.keySet()) {
					if (doc1 < doc2) {
						numComputed++;						
						if (numComputed%25000==0) log.debug("Number computed {}, average sim {}",numComputed,(totalSim)/numComputed);
						WordVector v1 = documents.get(doc1);
						WordVector v2 = documents.get(doc2);
						if (v1.getNumWords()<wordThreshold || v2.getNumWords()<wordThreshold) continue;
						
						double res = CosineSimilarity.cosineSimilarity(documents.get(doc1), documents.get(doc2));
						totalSim += res;
						if (res>=simThreshold) {
							//Write to file
							out.write(String.valueOf(doc1.intValue()));
							out.write("\t");
							out.write(String.valueOf(doc2.intValue()));
							out.write("\t");
							out.write(String.valueOf(res));
							out.write("\n");
						}
					}
				}
			}
	    	out.close();
		} catch (IOException e) {
	    	System.err.println("Error in writing file: " + e.getMessage());
	    	e.printStackTrace();
	    }
	}

}
