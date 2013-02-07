package edu.umd.cs.linqs.wiki;

import edu.umd.cs.psl.model.argument.GroundTerm;


public class GroundingWrapper implements Comparable<GroundingWrapper> {
	private GroundTerm [] grounding;

	public GroundingWrapper(GroundTerm [] args) {
		grounding = args;
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
		int code = 0;
		for (int i = 0; i < grounding.length; i++)
			code += 13*i*grounding[i].hashCode();
		return code;
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