package edu.umd.cs.psl.experiments.wikipedia;

import java.sql.*;
import java.util.StringTokenizer;
import java.io.*;

public class WikipediaDBCreator {
	private String dataPath;
	private int numFolds;
	
	private final Connection conn;
	
	public WikipediaDBCreator(String pathToDataFolder, int numFolds, Connection c){
		dataPath = pathToDataFolder;
		this.numFolds = numFolds;
		conn = c;
	}
	
	public void setUpDatabase() throws Exception {
		
		setUpDocument();
		setUpCategoryBelonging();
		setUpCategory();
		//setUpEditEvent(conn);
		//setUpUserTalk(conn);
		setUpBinaryEditEvent();
		setUpBinaryTopicTalk();
		//setUpBinaryUserTalk(conn);
		setUpWithinWithinLink();
		setUpWithinOutsideLink();
		//setUpPageLink(conn);
		//conn.close();
	}
	
	
	
	private void setUpDocument() throws Exception{
		String createTable = "create table document (docId int, docText mediumtext, docQuality int, foldId int);";
		createTable(createTable, "document");
		loadUpTable("Document.txt", "document", 4);
	}
	
	private void setUpCategoryBelonging() throws Exception{
		String createTable = "create table categorybelonging (docId int, catId int);";
		createTable(createTable, "categorybelonging");
		loadUpTable("CategoryBelonging.txt", "categorybelonging", 2);
	}
	
	private void setUpCategory() throws Exception{
		String createTable = "create table category (catId int, catName varchar(100));";
		createTable(createTable, "category");
		loadUpTable("Category.txt", "category", 2);
	}
	
	private void setUpEditEvent() throws Exception{
		String createTable = "create table editevent (docId int, uId int, firstEdit char(20), minor tinyint, topicTalk tinyint );";
		createTable( createTable, "editevent");
		loadUpTable("EditEvent.txt", "editevent", 5);
	}
	
	private void setUpUserTalk() throws Exception{
		String createTable = "create table usertalk (hostId int, contribId int, firstEdit char(20), minor tinyint);";
		createTable( createTable, "usertalk");
		loadUpTable("UserTalk.txt", "usertalk", 4);
	}
	
	private void setUpWithinWithinLink() throws Exception{
		String createTable = "create table withinwithinlink (fromDocId int, toDocId int);";
		createTable( createTable, "withinwithinlink");
		loadUpTable("WithinWithinLink.txt", "withinwithinlink", 2);
	}
	
	private void setUpWithinOutsideLink() throws Exception{
		String createTable = "create table withinoutsidelink (fromDocId int, toDocId int);";
		createTable( createTable, "withinoutsidelink");
		loadUpTable("WithinOutsideLink.txt", "withinoutsidelink", 2);
	}
	
	private void setUpPageLink() throws Exception{
		String createTable = "create table pagelink (docId int, url varchar(255));";
		createTable(createTable, "pagelink");
		loadUpTable("PageLink.txt", "pagelink", 2);
	}
	
	private void setUpBinaryEditEvent() throws Exception{
		String createTable = "create table editevent (docId int, uId int);";
		createTable( createTable, "editevent");
		loadUpTable("BinaryEditEvent.txt", "editevent",  2);
	}
	
	private void setUpBinaryTopicTalk() throws Exception{
		String createTable = "create table topictalk (docId int, uId int);";
		createTable( createTable, "topictalk");
		loadUpTable("BinaryTopicTalk.txt", "topicTalk",  2);
	}
	
	private void setUpBinaryUserTalk() throws Exception{
		String createTable = "create table usertalk (hostId int, contribId int);";
		createTable( createTable, "usertalk");
		loadUpTable("BinaryUserTalk.txt", "usertalk",  2);
	}
	
	private void createTable(String creationCommand, String tableName) throws Exception{
		Statement stat = conn.createStatement();
		stat.executeUpdate("drop table if exists " + tableName + ";");
		stat.executeUpdate(creationCommand);
		stat.close();
	}
	
	private void loadUpTable(String tblFN, String tblName, int numValues) throws Exception{
		StringBuffer statementStr = new StringBuffer();
		statementStr.append("insert into ");
		statementStr.append(tblName);
		statementStr.append(" values (?");
		for (int i = 0; i < numValues - 1; i++){
			statementStr.append(", ?");
		}
		statementStr.append(");");
		
		PreparedStatement prep = conn.prepareStatement(statementStr.toString());
		
		BufferedReader in  = new BufferedReader(new FileReader(this.dataPath + "/" + tblFN));
		String line = "";
		int foldCounter = 1;
		while ((line = in.readLine()) != null){
			StringTokenizer st = new StringTokenizer(line, "\t");
			int numTokens = st.countTokens();
		
			if (numTokens != numValues){ //the folds column
				assert (numValues == numTokens + 1); 
			}
			for (int i = 0; i < numTokens; i++){
				String nextToken = st.nextToken();
				prep.setString(i + 1, nextToken);
			}
			if (numTokens != numValues){
				prep.setString(numValues, Integer.toString(foldCounter));
				foldCounter++;
				if (foldCounter > this.numFolds){
					foldCounter = 1;
				}
			}
			prep.addBatch();
		}
		
		conn.setAutoCommit(false);
		prep.executeBatch();
		conn.setAutoCommit(true);
	}
	/**
	 * @param args
	 */
	public static void main(String[] args) throws Exception{
		Class.forName("org.sqlite.JDBC");
		Connection conn = DriverManager.getConnection("jdbc:sqlite:test.db");

		WikipediaDBCreator creator = new WikipediaDBCreator("/fs/cocoabeach/lily/WikipediaProcessedData/DataForPSL/TestTables", 5, conn);
		creator.setUpDatabase();
		conn.close();
	}

}
