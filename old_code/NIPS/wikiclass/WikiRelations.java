package edu.umd.cs.psl.experiments.wikiclass;

import edu.umd.cs.psl.ui.data.graph.RelationType;

public enum WikiRelations implements RelationType {

	Link {

		@Override
		public int arity() {
			return 2;
		}

		@Override
		public boolean isSoft() {
			return false;
		}

		@Override
		public boolean isSymmetric() {
			return false;
		}
		
		@Override
		public boolean hasAttributes() {
			return false;
		}
		
	},
	
	Edit {
		@Override
		public int arity() {
			return 2;
		}

		@Override
		public boolean isSoft() {
			return true;
		}

		@Override
		public boolean isSymmetric() {
			return false;
		}
		
		@Override
		public boolean hasAttributes() {
			return true;
		}
	},
	
	Talk {
		@Override
		public int arity() {
			return 2;
		}

		@Override
		public boolean isSoft() {
			return true;
		}

		@Override
		public boolean isSymmetric() {
			return false;
		}		
		
		@Override
		public boolean hasAttributes() {
			return true;
		}

	},
	
	TalksTo {
		@Override
		public int arity() {
			return 2;
		}

		@Override
		public boolean isSoft() {
			return true;
		}

		@Override
		public boolean isSymmetric() {
			return false;
		}		
		
		@Override
		public boolean hasAttributes() {
			return true;
		}

	},
	
	HasCategory {
		@Override
		public int arity() {
			return 2;
		}

		@Override
		public boolean isSoft() {
			return false;
		}

		@Override
		public boolean isSymmetric() {
			return false;
		}
		
		@Override
		public boolean hasAttributes() {
			return false;
		}

	},
	
	SimilarText {
		@Override
		public int arity() {
			return 2;
		}

		@Override
		public boolean isSoft() {
			return true;
		}

		@Override
		public boolean isSymmetric() {
			return true;
		}	
		
		@Override
		public boolean hasAttributes() {
			return true;
		}

	};
}
