// -*- C++ -*-
#ifndef ESSENTIALBOUNDARYCONDITIONS_H
#define ESSENTIALBOUNDARYCONDITIONS_H

#include "Utilities.h"

class EssentialBC {
public:
	EssentialBC() {}
	EssentialBC(const size_t & nodeId, 
		const size_t & direction, const double & value, const size_t & flag) {
		_nodeId = nodeId;
		_direction = direction;
		_value = value;
		_flag = flag;
	}

	size_t
	getNodeId() const {
		return _nodeId;
	}

	size_t
	getDirection() const {
		return _direction;
	}

	double
	getValue() const {
		return _value;
	}

	size_t
	getFlag() const {
		return _flag;
	}

	void
	changeBoundaryValue (const double & value) {
		_value = value;
	}

	void
	scaleBoundaryValue (const double & scale) {
		_value *= scale;
	}
private:
	size_t _nodeId;
	size_t _direction;
	double _value;
	size_t _flag;  // 0 means no need to update, 1 means need update
};

#endif  // ESSENTIALBOUNDARYCONDITIONS_H
