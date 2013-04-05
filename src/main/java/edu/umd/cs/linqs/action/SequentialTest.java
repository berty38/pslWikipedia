package edu.umd.cs.linqs.action;

import edu.umd.cs.psl.database.ReadOnlyDatabase;
import edu.umd.cs.psl.model.argument.ArgumentType;
import edu.umd.cs.psl.model.argument.GroundTerm;
import edu.umd.cs.psl.model.argument.IntegerAttribute;
import edu.umd.cs.psl.model.function.ExternalFunction;

public class SequentialTest implements ExternalFunction {

	private static final ArgumentType[] argTypes = new ArgumentType[]{ ArgumentType.Integer, ArgumentType.Integer };
	
	@Override
	public double getValue(ReadOnlyDatabase db, GroundTerm... args) {
		int t1 = ((IntegerAttribute) args[0]).getValue().intValue();
		int t2 = ((IntegerAttribute) args[1]).getValue().intValue();
		if (Math.abs(t1-t2) == 1)
			return 1.0;
		return 0.0;
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
