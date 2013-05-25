#!/bin/bash
mvn compile

mvn dependency:build-classpath -Dmdep.outputFile=classpath.out

for type in "left" "bottom"
do
	for train in "rand"
	do
		java -Xmx45g -cp ./target/classes:`cat classpath.out` edu.umd.cs.linqs.vision.OlivettiTest $type $train| tee -a olivettiLog.txt
	done
done