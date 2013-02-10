package edu.umd.cs.linqs.wiki; 

import java.io.FileReader;
import java.io.IOException;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Scanner;
import java.util.Set;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import edu.umd.cs.psl.database.loading.Inserter;


/**
 * This class takes a set of unique keys, reads a file written in sparse format 
 * of word counts (or tf-idf scores) and category labels and learns a Naive Bayes 
 * classifier for the labels.
 * 
 * @author Bert Huang <bert@cs.umd.edu>
 *
 */
public class NaiveBayesUtil {

	private Logger log;
	private Set<Integer> allWords;
	private CounterMap<Integer> catDocTotals;
	private CounterMap<Integer> totalWordsPerCat;
	private Map<Integer, CounterMap<Integer>> wordsPerCat;
	

	private final int prior = 1;
	private final int catPrior = 1;

	public NaiveBayesUtil() {
		log = LoggerFactory.getLogger("NaiveBayesUtility");
		allWords = new HashSet<Integer>();
	}


	/**
	 * Loads word count data and labels
	 * @param trainingKeys set of training unique keys
	 */
	public void learn(Set<Integer> trainingKeys, String categoryFile, String wordFile) {
		try {
			Scanner catScanner = new Scanner(new FileReader(categoryFile));

			catDocTotals = new CounterMap<Integer>();
			totalWordsPerCat = new CounterMap<Integer>();

			Map<Integer, Integer> categories = new HashMap<Integer, Integer>();

			int documentCount = 0;

			while (catScanner.hasNext()) {
				String line = catScanner.nextLine();
				String [] tokens = line.split("\t");
				if (tokens.length == 2) {
					Integer docID = Integer.decode(tokens[0]);
					Integer label = Integer.decode(tokens[1]);

					if (trainingKeys.contains(docID)) {
						categories.put(docID, label);

						catDocTotals.increment(label);
						documentCount++;
					}
				}
			}
			catScanner.close();

			// initialize category word counter
			wordsPerCat = new HashMap<Integer, CounterMap<Integer>>();
			for (Integer cat : catDocTotals.keySet())  {
				wordsPerCat.put(cat, new CounterMap<Integer>());
			}

			// load words
			log.debug("Starting word counts");
			Scanner wordScanner = new Scanner(new FileReader(wordFile));
			while (wordScanner.hasNext()) {
				String line = wordScanner.nextLine();
				String [] tokens = line.split("\t");
				Integer docID = Integer.decode(tokens[0]);
				Map<Integer, Double> docWords = parseWords(tokens[1]);
				for (Map.Entry<Integer, Double> e : docWords.entrySet()) {
					Integer wordID = e.getKey();
					Double count = (double) e.getValue();
					if (trainingKeys.contains(docID)) {
						Integer cat = categories.get(docID);
						wordsPerCat.get(cat).increment(wordID, count);
						totalWordsPerCat.increment(cat, count);
						//log.debug("Document {} in category {} had word " + wordID, docID, cat);
					}
					allWords.add(wordID);
				}
			}
			wordScanner.close();

			log.debug("Finished word counts");

			if (log.isDebugEnabled()) {
				// evaluate training accuracy
				int correct = 0;
				int total = 0;
				wordScanner = new Scanner(new FileReader(wordFile));
				while (wordScanner.hasNext()) {
					String line = wordScanner.nextLine();
					String [] tokens = line.split("\t");
					Integer docID = Integer.decode(tokens[0]);
					if (categories.containsKey(docID)) {
						Integer cat = categories.get(docID);
						Map<Integer, Double> docWords = parseWords(tokens[1]);
						Map<Integer, Double> prediction = predict(docWords);
						Integer pred = -1;
						//log.debug(prediction.toString());
						for (Integer c : wordsPerCat.keySet())
							if (pred == -1 || prediction.get(c) > prediction.get(pred))
								pred = c;
						//log.debug("Predicted {}, true {}", pred, cat);
						if (cat.equals(pred))
							correct++;
						total++;
					}
				}
				wordScanner.close();

				log.debug("Naive Bayes training accuracy: {}", (double) correct / (double) total);
			}

			categories.clear();
		} catch (IOException e) {
			log.error(e.toString());
		}
	}

	/**
	 * Takes a map of unnormalized log-likelihoods and normalizes them in a 
	 * numerically stable way
	 * @param scores
	 */
	private void convertToProbability(Map<Integer, Double> scores) {
		double max = Double.NEGATIVE_INFINITY;
		for (Double score : scores.values())
			max = Math.max(max, score);
		double normalizer = 0;
		for (Double score : scores.values())
			normalizer += Math.exp(score - max);
		for (Map.Entry<Integer, Double> e : scores.entrySet()) 
			scores.put(e.getKey(), Math.exp(e.getValue() - max - Math.log(normalizer)));
	}

	private Map<Integer, Double> parseWords(String string) {
		String [] tokens = string.split(" ");
		Map<Integer, Double> words = new HashMap<Integer, Double>(1000);
		for (int i = 0; i < tokens.length; i++) {
			if (tokens[i].length() > 1) {
				String [] subTokens = tokens[i].split(":");
				words.put(Integer.decode(subTokens[0]), Double.parseDouble(subTokens[1]));
			}
		}
		return words;
	}

	/**
	 * returns a map of the probabilities for each category
	 * @param docWords word ids contained in this file
	 * @return map of probabilities for each category
	 */
	public Map<Integer, Double> predict(Map<Integer, Double> docWords) {
		Map<Integer, Double> scores = new HashMap<Integer, Double>();
		for (Integer cat : catDocTotals.keySet()) {
			double score = Math.log(catDocTotals.get(cat) + catPrior);
			CounterMap<Integer> counts = wordsPerCat.get(cat);
			for (Map.Entry<Integer, Double> e : docWords.entrySet()) {
				Integer word = e.getKey();
				Double count = e.getValue();
				if (allWords.contains(word)) {
					score += count * (Math.log(counts.get(word) + prior) - Math.log(totalWordsPerCat.get(cat) + allWords.size()*prior));
				}
			}
			//log.debug("Document scores " + score + " for label " + cat);
			scores.put(cat, score);
		}
		convertToProbability(scores);
		return scores;
	}
	
	/**
	 * Predicts the maximum likelihood assignment 
	 * @param docWords set of observed words
	 * @return category with maximum likelihood
	 */
	public Integer predictBest(Map<Integer, Double> docWords) {
		Map<Integer, Double> scores = predict(docWords);
		double bestScore = Double.NEGATIVE_INFINITY;
		Integer bestCat = 0;
		for (Map.Entry<Integer, Double> e : scores.entrySet()) {
			if (e.getValue() > bestScore) {
				bestScore = e.getValue();
				bestCat = e.getKey();
			}
		}
		return bestCat;
	}

	/**
	 * reads in a file of word counts or tf-idf scores and generates the 
	 * predicted label for each document
	 * @param wordFile filename of document containing word counts
	 * @return map between document ID and predicted label
	 */
	public Map<Integer, Map<Integer, Double>> predictAll(String wordFile, Set<Integer> documents) {
		Map<Integer, Map<Integer, Double>> predictions = new HashMap<Integer, Map<Integer, Double>>();
		// load words
		try {
			log.trace("Starting to load file for prediction");
			Scanner wordScanner = new Scanner(new FileReader(wordFile));
			while (wordScanner.hasNext()) {
				String line = wordScanner.nextLine();
				String [] tokens = line.split("\t");
				Integer docID = Integer.decode(tokens[0]);
				if (documents.contains(docID)) {
					Map<Integer, Double> docWords = parseWords(tokens[1]);

					predictions.put(docID, this.predict(docWords));
					docWords.clear();
				}
				//log.trace("Document " + docID + ", prediction: " + predictions.get(docID));
			}
			wordScanner.close();
			log.trace("Finished prediction on full file");
		} catch (IOException e) {
			log.error(e.toString());
		}
		return predictions;
	}

	/**
	 * reads in a file of word counts or tf-idf scores and generates the 
	 * probability for each label of each document
	 * @param wordFile filename of document containing word counts
	 * @return map between document ID and predicted label
	 */
	public void insertAllProbabilities(String wordFile, Set<Integer> documents, Inserter inserter) {
		// load words
		try {
			log.debug("Loading file for Naive Bayes prediction");
			Scanner wordScanner = new Scanner(new FileReader(wordFile));
			int count = 0;
			while (wordScanner.hasNext()) {
				String line = wordScanner.nextLine();
				String [] tokens = line.split("\t");
				Integer docID = Integer.decode(tokens[0]);
				if (documents.contains(docID)) {
					Map<Integer, Double> docWords = parseWords(tokens[1]);

					Map<Integer, Double> probs = this.predict(docWords);

					for (Map.Entry<Integer, Double> e : probs.entrySet()) {
						inserter.insertValue(e.getValue(), docID, e.getKey());
						//log.trace("NB predicts p={} for {}", e.getValue(), e.getKey());
					}
					count++;
				}
			}
			wordScanner.close();
			log.debug("Finished prediction on full file. Inserted {} documents", count);
		} catch (IOException e) {
			log.error(e.toString());
		}
	}
	
	/**
	 * reads in a file of word counts or tf-idf scores and generates the 
	 * predicted label for each document
	 * @param wordFile filename of document containing word counts
	 * @return map between document ID and predicted label
	 */
	public void insertAllPredictions(String wordFile, Set<Integer> documents, Inserter inserter) {
		// load words
		try {
			log.debug("Loading file for Naive Bayes prediction");
			Scanner wordScanner = new Scanner(new FileReader(wordFile));
			while (wordScanner.hasNext()) {
				String line = wordScanner.nextLine();
				String [] tokens = line.split("\t");
				Integer docID = Integer.decode(tokens[0]);
				if (documents.contains(docID)) {
					Map<Integer, Double> docWords = parseWords(tokens[1]);

					Integer cat = this.predictBest(docWords);
					
					inserter.insertValue(1.0, docID, cat);
				}
			}
			wordScanner.close();
			log.trace("Finished prediction on full file");
		} catch (IOException e) {
			log.error(e.toString());
		}
	}

	
	private class CounterMap<T> {
		public CounterMap() {
			counts = new HashMap<T, Double>();
		}

		public void increment(T key) {
			this.increment(key, 1);
		}
		
		public void increment(T key, Number i) {
			double c = 0;
			if (counts.containsKey(key))
				c = counts.get(key);
			counts.put(key, i.doubleValue() + c);
		}

		public int get(T key) {
			if (counts.containsKey(key))
				return counts.get(key).intValue();
			return 0;
		}

		public void clear() { counts.clear(); };
		public Set<T> keySet() { return counts.keySet(); }
		public Set<Map.Entry<T, Double>> entrySet() { return counts.entrySet(); }

		private Map<T,Double> counts;

	}
}
