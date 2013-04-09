package edu.umd.cs.linqs.action;

import edu.umd.cs.psl.database.ReadOnlyDatabase;
import edu.umd.cs.psl.model.argument.ArgumentType;
import edu.umd.cs.psl.model.argument.GroundTerm;
import edu.umd.cs.psl.model.argument.IntegerAttribute;
import edu.umd.cs.psl.model.function.ExternalFunction;

class ClosenessFunction implements ExternalFunction {

	public static final double DEFAULT_SIGMA = 1;
	public static final double DEFAULT_THRESH = 1e-2;
	
	private static final ArgumentType[] argTypes = new ArgumentType[]{
		ArgumentType.Integer,ArgumentType.Integer,	// x-coord
		ArgumentType.Integer,ArgumentType.Integer,	// y-coord
		ArgumentType.Integer,ArgumentType.Integer,	// width
		ArgumentType.Integer,ArgumentType.Integer	// height
		};
	
	private final double sigma;
	private final double thresh;
	
	public ClosenessFunction() {
		this(DEFAULT_SIGMA, DEFAULT_THRESH);
	}
	
	public ClosenessFunction(final double sigma, final double thresh) {
		this.sigma = sigma;
		this.thresh = thresh;
	}
	
	@Override
	public double getValue(ReadOnlyDatabase db, GroundTerm... args) {
		/* Get args */
		int x1 = ((IntegerAttribute) args[0]).getValue().intValue();
		int x2 = ((IntegerAttribute) args[1]).getValue().intValue();
		int y1 = ((IntegerAttribute) args[2]).getValue().intValue();
		int y2 = ((IntegerAttribute) args[3]).getValue().intValue();
		int w1 = ((IntegerAttribute) args[4]).getValue().intValue();
		int w2 = ((IntegerAttribute) args[5]).getValue().intValue();
		int h1 = ((IntegerAttribute) args[6]).getValue().intValue();
		int h2 = ((IntegerAttribute) args[7]).getValue().intValue();

		//TODO: modify distance function to something more sophisticated
		double dx = Math.abs(x1-x2);
		double dy = Math.abs(y1-y2);
		double v = Math.exp(-(dx*dx + dy*dy) / sigma);
		
		return v < thresh ? 0.0 : v;
	}

	@Override
	public int getArity() {
		return argTypes.length;
	}

	@Override
	public ArgumentType[] getArgumentTypes() {
		return argTypes;
	}

}

