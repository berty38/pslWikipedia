package edu.umd.cs.linqs.wiki;

import edu.umd.cs.psl.model.argument.GroundTerm;
import edu.umd.cs.psl.model.atom.GroundAtom;

import org.apache.commons.lang.builder.HashCodeBuilder;


public class GroundingWrapper implements Comparable<GroundingWrapper> {
	private GroundTerm [] grounding;
	private final int hashcode;
	
	public GroundingWrapper(GroundTerm [] args) {
		grounding = args;
		
		HashCodeBuilder hcb = new HashCodeBuilder();
		for (GroundTerm term : grounding)
			hcb.append(term);
		
		hashcode = hcb.toHashCode();
	}

	public GroundTerm [] getArray() { return grounding; }

	@Override
	public boolean equals(Object oth) {
		GroundingWrapper gw = (GroundingWrapper) oth;
		if (gw.getArray().length != grounding.length)
			return false;
		for (int i = 0; i < grounding.length; i++)
			if (!gw.getArray()[i].equals(grounding[i]))
				return false;
		return true;
	}
	
	@Override
	public int hashCode() {
		return hashcode;
	}

	@Override
	public int compareTo(GroundingWrapper oth) {
		if (grounding.length < oth.grounding.length)
			return -1;
		else if (grounding.length > oth.grounding.length)
			return 1;
		else {
			for (int i = 0; i < grounding.length; i++) {
				int elementCompare = grounding[i].compareTo(oth.grounding[i]);
				if (elementCompare != 0)
					return elementCompare;
			}
		}
		return 0;
	}

}