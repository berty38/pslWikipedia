#!/bin/bash
mvn compile

mvn dependency:build-classpath -Dmdep.outputFile=classpath.out

java -Xmx10000m -cp ./target/classes:`cat classpath.out` edu.umd.cs.linqs.wiki.WikiCategoryTestNB | tee -a wikiLog.txt
