#!/bin/bash
mvn compile

mvn dependency:build-classpath -Dmdep.outputFile=classpath.out

java -Xmx12g -cp ./target/classes:`cat classpath.out` edu.umd.cs.linqs.jester.Jester
