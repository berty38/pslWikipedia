package edu.umd.cs.psl.experiments.wikipedia;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.HashSet;
import java.util.Set;
import java.util.StringTokenizer;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import de.mathnbits.statistics.IntegerDist;
import de.mathnbits.statistics.ObjectCounter;

import edu.umd.cs.psl.database.loading.DataLoader;
import edu.umd.cs.psl.database.loading.OpenInserter;
import edu.umd.cs.psl.ui.experiment.folds.MultiFoldLoader;

public class WikipediaDataLoader implements MultiFoldLoader {
	
	private static final Logger log = LoggerFactory.getLogger(WikipediaDataLoader.class);
	
	public static int testFoldId = 4;
	public static int testGroundTruthFoldId = 5;
	public static int trainFoldId = 1;
	public static int trainGroundTruthFoldId = 2;
	public static int trainFoldRemoveId = 3;
	
	private static final int leaveoutevery = 2;
	private static final double editNormalizer=10.0;
	
	private final String dataPath;
	private final int numFolds;
	
	private int currFoldId;
	private HashSet<Integer> withheldDocuments;
	private int withinTrainEvidenceCounter; //for maintaining during training the correct proportion of unknowns
	private HashSet<Integer> unknownInTrainDocuments, knownInTrainDocuments;
	
	private HashSet<Integer> interestingDocuments, interestingPeople;
	private HashSet <Integer> activeDocs;
		
	public WikipediaDataLoader(String pathToData, int _numFolds){
		dataPath = pathToData;
		numFolds = _numFolds;
		currFoldId = -1;
		withinTrainEvidenceCounter = 0;
		
		withheldDocuments = new HashSet<Integer>();
		unknownInTrainDocuments = new HashSet<Integer>();
		knownInTrainDocuments = new HashSet<Integer>();
		
		activeDocs = new HashSet<Integer>();
		interestingDocuments = new HashSet<Integer>();
		interestingPeople = new HashSet<Integer>();
	}
	
	@Override
	public int[] getTestDataGroundTruthIDs() {
		int [] result = {WikipediaDataLoader.trainFoldId, WikipediaDataLoader.trainGroundTruthFoldId, WikipediaDataLoader.testFoldId, WikipediaDataLoader.testGroundTruthFoldId};
		return result; //WikipediaDataLoader.testGroundTruthFoldId;
	}

	@Override
	public int[] getTestDataIDs() {
		int [] result = {WikipediaDataLoader.trainFoldId, WikipediaDataLoader.trainGroundTruthFoldId, WikipediaDataLoader.testFoldId};
		return result; //WikipediaDataLoader.testFoldId;
	}

	@Override
	public int[] getTrainDataGroundTruthIDs() {
		int [] result = {WikipediaDataLoader.trainFoldId, WikipediaDataLoader.trainFoldRemoveId, WikipediaDataLoader.trainGroundTruthFoldId};
		return result; //WikipediaDataLoader.trainGroundTruthFoldId;
	}

	@Override
	public int[] getTrainDataIDs() {
		int[] result = {WikipediaDataLoader.trainFoldId, WikipediaDataLoader.trainFoldRemoveId} ;// new int[2];
		return result; //WikipediaDataLoader.trainFoldId;
	}

	@Override
	public boolean hasNextFold() {
		//folds range from 0 to numFolds -1
		return currFoldId < numFolds - 1;
	}

	@Override
	public void loadNextFold(DataLoader loader){
		currFoldId++;
		withheldDocuments.clear();
		unknownInTrainDocuments.clear();
		
		try{
		this.getActiveDocs();
		this.collectInterestingPeopleAndDocuments();
		makeDocumentTable(loader);
		
		makeDocIndepTable("category", "Category.txt", loader);
		//makeUserTable(loader);
		makeUserTalkTable(loader);
		boolean targetTable = false;
		boolean considerBothColumns = false;
		//binaryNoMinorsTwoWeeksEditEvent , BinaryTwoWeeksEditEvent.txt
		//makeDocDepTable("editevent", "binaryNoMinorsTwoWeeksEditEvent.txt", targetTable, considerBothColumns, loader, interestingDocuments, interestingPeople );
		makeUserEditTable("editevent", "twoYearEditEventCounts.txt", loader);
		
		//makeDocDepTable("topictalk", "BinaryTopicTalk.txt", targetTable, considerBothColumns, loader);
		//makeDocDepTable("pagelink", "PageLink.txt", targetTable, considerBothColumns, loader);
		//makeDocDepTable("withinoutsidelink", "WithinOutsideLink.txt", targetTable, considerBothColumns, loader, interestingDocuments, null);
		considerBothColumns = true;
		makeDocDepTable("withinwithinlink", "WithinWithinLink.txt", targetTable, considerBothColumns, loader, interestingDocuments, interestingDocuments);
		considerBothColumns = false;
		targetTable = true;
		makeDocDepTable("categorybelonging", "CategoryBelonging.txt", targetTable, considerBothColumns, loader, interestingDocuments, null);
		makeUnknownAndKnownTables(loader);
		}catch (Exception e){
			e.printStackTrace();
		}	
	}

	/*
	 * targetTable: is it a target table whose values we'll be predicting
	 * considerBothColumns: do both columns contain documents, or is it just the first one
	 */
	private void makeDocDepTable(String tableName, String fileName,
			boolean targetTable, boolean considerBothColumns,
			DataLoader loader, Set<Integer> pruneFirst, Set<Integer> pruneSecond) throws Exception{
		OpenInserter ih = loader.getOpenInserter(tableName);
		OpenInserter ihp = loader.getOpenInserter("equalPage");
		Set<Integer> pages = new HashSet<Integer>();
		
		BufferedReader in  = new BufferedReader(new FileReader(this.dataPath + "/" + fileName));
		String line = "";
		while ((line = in.readLine()) != null){
			StringTokenizer st = new StringTokenizer(line, "\t");
			if (st.countTokens() != 2) continue;
			
			String first = st.nextToken();
			String second = st.nextToken();
			
			Integer docId1 = new Integer(first);
			Integer docId2 = new Integer(second);
			
			if (pruneFirst!=null && !pruneFirst.contains(docId1)) continue;
			if (pruneSecond!=null && !pruneSecond.contains(docId2)) continue;

			
			if (tableName.equals("withinoutsidelink")) {
				if (!pages.contains(docId2)) {
					ihp.insert(WikipediaDataLoader.trainFoldId, second, second);
					pages.add(docId2);
				}
			}
			
			boolean isTest = (this.withheldDocuments.contains(new Integer(first)));
			if (!isTest && considerBothColumns)
				isTest = (this.withheldDocuments.contains(new Integer(second)));
			
			int testTrain = 0;
			if (isTest){
				if(targetTable) testTrain = WikipediaDataLoader.testGroundTruthFoldId;
				else testTrain = WikipediaDataLoader.testFoldId;
			}
			else{ //train fold


				
				
				
				if (targetTable) {
					// some of these cases are ground truth and some are evidence, 
					// proportion needs to be as in test data, 1 in numFolds
					this.withinTrainEvidenceCounter++;
					if (this.withinTrainEvidenceCounter % numFolds == 0){
						testTrain = WikipediaDataLoader.trainGroundTruthFoldId;
						this.unknownInTrainDocuments.add(new Integer(first));
					}
					else{
						testTrain = WikipediaDataLoader.trainFoldId;
						this.knownInTrainDocuments.add(new Integer(first));
					}
				}
				else testTrain = WikipediaDataLoader.trainFoldId;
			}
			
			ih.insert(testTrain, first, second);
		}
				
	}

	//Takes care of tables that do not talk about documents
	private void makeDocIndepTable(String tableName, String fileName,
			DataLoader loader) throws Exception{
		OpenInserter ih = loader.getOpenInserter(tableName);
		
		BufferedReader in  = new BufferedReader(new FileReader(this.dataPath + "/" + fileName));
		String line = "";
		while ((line = in.readLine())!= null){
			StringTokenizer st = new StringTokenizer(line, "\t");
			if (st.countTokens() != 2) continue;
			String first = st.nextToken();
			String second = st.nextToken();
			
			ih.insert(WikipediaDataLoader.trainFoldId, first, second);
		}
	}
	
	private void makeUserEditTable(String tableName, String fileName, DataLoader loader) throws Exception{
		OpenInserter ih = loader.getOpenInserter(tableName);
		
		BufferedReader in  = new BufferedReader(new FileReader(this.dataPath + "/" + fileName));
		String line = "";
		while ((line = in.readLine())!= null){
			StringTokenizer st = new StringTokenizer(line, "\t");
			if (st.countTokens() != 3) continue;
			String first = st.nextToken();
			String second = st.nextToken();
			String third = st.nextToken();
			Integer docid = new Integer(first);
			Integer userid = new Integer(second);
			Integer count = new Integer(third);
			
			if (!this.interestingDocuments.contains(docid) || !this.interestingPeople.contains(userid)) continue;
			
			if (count>3 && count<10)
				ih.insertTruth(WikipediaDataLoader.trainFoldId, Math.min(1.0, count/10.0), docid,userid);
		}
	}
	
	private void makeUserTalkTable(DataLoader loader) throws Exception{
		String fileName = "BinaryTwoWeeksUserTalk.txt";
		String tableName = "usertalk";
		OpenInserter ih = loader.getOpenInserter(tableName);
		
		BufferedReader in  = new BufferedReader(new FileReader(this.dataPath + "/" + fileName));
		String line = "";
		while ((line = in.readLine())!= null){
			StringTokenizer st = new StringTokenizer(line, "\t");
			if (st.countTokens() != 2) continue;
			String first = st.nextToken();
			String second = st.nextToken();
			Integer uid1 = new Integer(first);
			Integer uid2 = new Integer(second);
			
			if (!this.interestingPeople.contains(uid2) || !this.interestingPeople.contains(uid1)) continue;
			
			ih.insert(WikipediaDataLoader.trainFoldId, first, second);
		}
	}

	private void makeDocumentTable(DataLoader loader) throws Exception{
		OpenInserter ih = loader.getOpenInserter("document");
		
		BufferedReader in  = new BufferedReader(new FileReader(this.dataPath + "/Document.txt"));
		String line = "";
		int counter = 0;
		while ((line = in.readLine()) != null){
			StringTokenizer st = new StringTokenizer(line, "\t");
			if (st.countTokens() != 3) continue;
			String docId = st.nextToken();
			Integer doc = new Integer(docId);
			if (!this.interestingDocuments.contains(doc)) continue;

			String text = st.nextToken();
			String quality = st.nextToken();
			Integer qual = new Integer(quality);
			/*if (qual != 0) {
				this.interestingDocuments.remove(doc);
				continue;
			}*/
			
			counter++;
			int testTrain = WikipediaDataLoader.trainFoldId;
			if (counter % numFolds == currFoldId){
				testTrain = WikipediaDataLoader.testFoldId;
				this.withheldDocuments.add(new Integer(docId));	
			}
			if (counter%leaveoutevery!=1)
				ih.insert(testTrain, docId, text, quality);		
		}
	}
	
	private void getActiveDocs() throws Exception{		
		BufferedReader in  = new BufferedReader(new FileReader(this.dataPath + "/Document.txt"));
		String line = "";
		while ((line = in.readLine()) != null){
			StringTokenizer st = new StringTokenizer(line, "\t");
			if (st.countTokens() != 3) continue;
			String docId = st.nextToken();
			String text = st.nextToken();
			String quality = st.nextToken();
			Integer qual = new Integer(quality);
			if (qual == 1) {
				this.activeDocs.add(new Integer(docId));
				continue;
			}		
		}
	}
	
	private void makeUnknownAndKnownTables(DataLoader loader){
		OpenInserter ihUnknown = loader.getOpenInserter("unknown");
		OpenInserter ihKnown = loader.getOpenInserter("known");
		for (Integer i : this.withheldDocuments){
			ihUnknown.insert(WikipediaDataLoader.testFoldId, i);
		}
		for (Integer i : this.unknownInTrainDocuments){
			//ih.insert(WikipediaDataLoader.trainFoldId, i);
			ihUnknown.insert(WikipediaDataLoader.trainFoldRemoveId, i);
			ihKnown.insert(WikipediaDataLoader.testFoldId, i);
		}
		for (Integer i : this.knownInTrainDocuments){	
			ihKnown.insert(WikipediaDataLoader.trainFoldId, i);
		}
	}
	
	private void makeUserTable(DataLoader loader){
		OpenInserter ihuser = loader.getOpenInserter("similarUsers");
		for (Integer i : interestingPeople){
			ihuser.insert(WikipediaDataLoader.trainFoldId,i, i);
		}
	}
	
	private void collectInterestingPeopleAndDocuments() throws Exception{
		BufferedReader in  = new BufferedReader(new FileReader(this.dataPath + "/BinaryTwoWeeksEditEvent.txt"));
		String line = "";
		
		ObjectCounter<Integer> userDegree = new ObjectCounter<Integer>();
		
		while ((line = in.readLine()) != null){
			StringTokenizer st = new StringTokenizer(line, "\t");
			Integer docId = new Integer(st.nextToken());
			if (!this.activeDocs.contains(docId))continue;
			Integer uId = new Integer(st.nextToken());
			this.interestingDocuments.add(docId);
			this.interestingPeople.add(uId);
			userDegree.increase(uId);
		}
		
//		IntegerDist degree = new IntegerDist();
		for (Integer i : userDegree.getObjects()) {
			if (userDegree.getCount(i)<2 || userDegree.getCount(i)>3) {
				if (!interestingPeople.remove(i)) throw new AssertionError();
			}
		}
		
//		log.debug("Mean degree {}, stdev {}",degree.mean(),degree.stdDev());
//		log.debug("Max {}, Num user {}",degree.max(),degree.numBins());
//		log.debug("{}",degree.toString());
	}
}
