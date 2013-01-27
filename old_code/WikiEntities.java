package edu.umd.cs.psl.experiments.wikiclass;

import edu.umd.cs.psl.ui.data.graph.EntityType;

public enum WikiEntities implements EntityType {
	
	Document {

		@Override
		public boolean hasAttributes() {
			return true;
		}
		
	},
	
	User {
		
		@Override
		public boolean hasAttributes() {
			return false;
		}
	},
	
	Category {
		
		@Override
		public boolean hasAttributes() {
			return true;
		}
	};

}
