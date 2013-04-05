/*
 * This file is part of the PSL software.
 * Copyright 2011 University of Maryland
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package edu.umd.cs.linqs.action;

import java.util.Iterator;
import java.util.List;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import cern.colt.Arrays;
import edu.umd.cs.psl.database.DataStore;
import edu.umd.cs.psl.database.Partition;
import edu.umd.cs.psl.database.loading.Inserter;
import edu.umd.cs.psl.model.predicate.Predicate;
import edu.umd.cs.psl.model.predicate.PredicateFactory;
import edu.umd.cs.psl.model.predicate.StandardPredicate;
import edu.umd.cs.psl.ui.data.file.util.DelimitedObjectConstructor;
import edu.umd.cs.psl.ui.data.file.util.LoadDelimitedData;

/**
 * Utility methods for common data-loading tasks.
 */
public class InserterUtils {
	
	private static final Logger log = LoggerFactory.getLogger(InserterUtils.class);

	public static void loadDelimitedDataMultiPartition(final Inserter insert, String file, String delimiter, int partitionColumn) {
		LoadDelimitedData.loadTabData(file, new DelimitedObjectConstructor<String>(){

			@Override
			public String create(String[] data) {
				//assert data.length==length;
				insert.insert((Object[])data);
				return null;
			}

			@Override
			public int length() {
				return 0;
			}
			
		}, delimiter);
	}
	
	public static void loadDelimitedDataMultiPartition(final Inserter insert, String file, int partitionColumn) {
		loadDelimitedDataMultiPartition(insert,file,LoadDelimitedData.defaultDelimiter, partitionColumn);
	}
	
	public static void loadDelimitedDataTruthMultiPartition(final Inserter insert, String file, String delimiter, int partitionColumn) {
		LoadDelimitedData.loadTabData(file, new DelimitedObjectConstructor<String>(){

			@Override
			public String create(String[] data) {
				double truth;
				try {
					truth = Double.parseDouble(data[data.length-1]);
				} catch (NumberFormatException e) {
					throw new AssertionError("Could not read truth value for data: " + Arrays.toString(data));
				}
				if (truth<0.0 || truth>1.0)
					throw new AssertionError("Illegal truth value encountered: " + truth);
				Object[] newdata = new Object[data.length-1];
				System.arraycopy(data, 0, newdata, 0, newdata.length);
				insert.insertValue(truth,newdata);
				return null;
			}

			@Override
			public int length() {
				return 0;
			}
			
		}, delimiter);
	}
	
	public static void loadDelimitedDataTruthMultiPartition(final Inserter insert, String file, int partitionColumn) {
		loadDelimitedDataTruthMultiPartition(insert,file,LoadDelimitedData.defaultDelimiter, partitionColumn);
	}

}
