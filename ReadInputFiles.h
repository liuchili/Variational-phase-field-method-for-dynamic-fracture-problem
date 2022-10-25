// -*- C++ -*-
#ifndef READINPUTFILES_H
#define READINPUTFILES_H

#include "WorldDynamic.h"
#include "EssentialBoundaryConditions.h" 

//namespace Utilities {
	vector<Vector2d>
	readNodePositions(string Positions) {
		string line_position;
		string partial;
		istringstream iss;

		ifstream file_position(Positions.c_str());
		getline(file_position, line_position);
		size_t numberOfNodes = atoi(line_position.c_str());
		//cout << "number of nodes is " << numberOfNodes << endl;

		Vector2d position;
		vector<Vector2d> nodePositions(numberOfNodes);
		for (unsigned int index = 0; index < numberOfNodes; index++) {
    	    getline(file_position, line_position);
    	    iss.str(line_position);
    	    getline(iss, partial, ' ');
    	    position(0) = atof(partial.c_str());
    	    getline(iss, partial, ' ');
    	    position(1) = atof(partial.c_str());
    	    nodePositions[index] = position;
    	    //cout << position(0) << " " << position(1) << endl;
    	    iss.clear();			
		}
		return nodePositions;
	}

	vector< vector<size_t> > 
	readElementMeshes(string Meshes) {
		string line_mesh;
		string partial;
		istringstream iss;

		ifstream file_mesh(Meshes.c_str());
		getline(file_mesh, line_mesh);
		size_t numberOfElements = atoi(line_mesh.c_str());
		//cout << "number of elements is " << numberOfElements << endl; 
		vector<size_t> elementmesh(3);
		vector< vector<size_t> > allElementMeshes(numberOfElements);

		for (unsigned int index = 0; index < numberOfElements; index++) {
			getline(file_mesh, line_mesh);
			iss.str(line_mesh);
			for (unsigned int eindex = 0; eindex < 3; eindex++) {
				getline(iss, partial, ' ');
				elementmesh[eindex] = atoi(partial.c_str());
			}
			iss.clear();
			allElementMeshes[index] = elementmesh;
		}
		return allElementMeshes;
	}

    
	vector< vector<size_t> >
	readBoundaryConnectivities(string Connectivities) {
		string line_connectivity;
		string partial;
		istringstream iss;

		ifstream file_connectivity(Connectivities.c_str());
		getline(file_connectivity, line_connectivity);
		size_t numberOfElements = atoi(line_connectivity.c_str());
		vector<size_t> singleConnection(2);
		vector< vector<size_t> > Connections(numberOfElements);

		for (unsigned int index = 0; index < numberOfElements; index++) {
			getline(file_connectivity, line_connectivity);
			iss.str(line_connectivity);
			for (unsigned int eindex = 0; eindex < 2; eindex++) {
				getline(iss, partial, ' ');
				singleConnection[eindex] = atoi(partial.c_str());
			}
			iss.clear();
			Connections[index] = singleConnection;
		}
		return Connections;		
	}

	vector<size_t>
	readBoundaryNodeIds(string NodeIds) {
		string line_id;
		ifstream file_id(NodeIds.c_str());
		getline(file_id, line_id);
		size_t numberOfNodes = atoi(line_id.c_str());
		vector<size_t> ids(numberOfNodes);
		for (unsigned int index = 0; index < numberOfNodes; index++) {
			getline(file_id, line_id);
			ids[index] = atoi(line_id.c_str());
		}
		return ids;
	}

	WorldDynamic
	generateWorldFromFile(string PositionsPhis, string Velocities, string Accelerations, string Meshes, 
		const double & k, const int & size) {
		string line_positionphi;
		string line_velocity;
		string line_acceleration;
		string line_mesh;

		string partial;
		istringstream iss;

		ifstream file_positionphi(PositionsPhis.c_str());
		ifstream file_velocity(Velocities.c_str());
		ifstream file_acceleration(Accelerations.c_str());
		ifstream file_mesh(Meshes.c_str());

		getline(file_positionphi, line_positionphi);
		const size_t numberOfNodes = atoi(line_positionphi.c_str());
		getline(file_mesh, line_mesh);
		const size_t numberOfElements = atoi(line_mesh.c_str());

		vector<Vector2d>          worldNodePositions(numberOfNodes);
		vector<Vector2d>          worldNodeVelocities(numberOfNodes);
		vector<Vector2d>          worldNodeAccelerations(numberOfNodes);
		vector<double>            worldNodePhis(numberOfNodes);
		vector<SimplexTriangle>   worldElements(numberOfElements);

		Vector2d                               position; position.fill(0.);
		Vector2d                               velocity; velocity.fill(0.);
		Vector2d                               acceleration; acceleration.fill(0.);
		double                                 phi = 0.;
		vector<size_t>                         singleElementNodeId(3);
		vector<Vector2d>                       singleElementNodePosition(3);
		vector<MaterialModelLinearElastic*>    singleElementMaterialModel;
		size_t                                 qpnumber = 0;
		double                                 lambda = 0.;
		double                                 nu = 0.;
		double                                 density = 0.;
		double                                 toughness = 0.;
		double                                 l0 = 0;
		size_t                                 flag = 0; // 0 means plane strain, 1 means plane stress


		for (unsigned int pindex = 0; pindex < numberOfNodes; pindex++) {
	        getline(file_positionphi, line_positionphi);
	        iss.str(line_positionphi);
	        getline(iss, partial, ' ');
	        position(0) = atof(partial.c_str());
	        getline(iss, partial, ' ');
	        position(1) = atof(partial.c_str());
	        getline(iss, partial, ' ');
	        phi = atof(partial.c_str());
	        worldNodePositions[pindex] = position;
	        worldNodePhis[pindex] = phi;
	        iss.clear();

	        getline(file_velocity, line_velocity);
	        iss.str(line_velocity);
	        getline(iss, partial, ' ');
	        velocity(0) = atof(partial.c_str());
	        getline(iss, partial, ' ');
	        velocity(1) = atof(partial.c_str());
	        worldNodeVelocities[pindex] = velocity;
	        iss.clear();

	        getline(file_acceleration, line_acceleration);
	        iss.str(line_acceleration);		
	        getline(iss, partial, ' ');
	        acceleration(0) = atof(partial.c_str());
	        getline(iss, partial, ' ');
	        acceleration(1) = atof(partial.c_str());
	        worldNodeAccelerations[pindex] = acceleration;
	        iss.clear();	        


		}
		for (unsigned int eindex = 0; eindex < numberOfElements; eindex++) {
			getline(file_mesh, line_mesh);
			iss.str(line_mesh);
			getline(iss, partial, ' ');
			singleElementNodeId[0] = atoi(partial.c_str());
			getline(iss, partial, ' ');
			singleElementNodeId[1] = atoi(partial.c_str());
			getline(iss, partial, ' ');
			singleElementNodeId[2] = atoi(partial.c_str());
			getline(iss, partial, ' ');
			qpnumber = atoi(partial.c_str());
			getline(iss, partial, ' ');
			lambda = atof(partial.c_str());
			getline(iss, partial, ' ');
			nu = atof(partial.c_str());
			getline(iss, partial, ' ');
			density = atof(partial.c_str());
			getline(iss, partial, ' ');
			toughness = atof(partial.c_str());
			getline(iss, partial, ' ');
			l0 = atof(partial.c_str());
			getline(iss, partial, ' ');
			flag = atoi(partial.c_str());
			iss.clear();

			//cout << "lambda, nu, density, toughness and flag is " << lambda << " " << nu << " " << density << " " << toughness << " " << flag << endl;

			singleElementMaterialModel.resize(qpnumber);
			MaterialModelLinearElastic* lelastic = new MaterialModelLinearElastic(lambda, nu, density, toughness, l0, flag);
			for (unsigned int qpindex = 0; qpindex < qpnumber; qpindex++) {
				singleElementMaterialModel[qpindex] = lelastic;
			}
			for (unsigned int index = 0; index < singleElementNodeId.size(); index++) {
				singleElementNodePosition[index] = worldNodePositions[singleElementNodeId[index]];
			}
			SimplexTriangle element = SimplexTriangle(singleElementNodePosition, singleElementNodeId, 
				singleElementMaterialModel, qpnumber);
			worldElements[eindex] = element;
		}
		WorldDynamic world = WorldDynamic(worldElements, numberOfNodes, worldNodePositions, worldNodeVelocities, worldNodeAccelerations, worldNodePhis, k, size);
		return world;
	}


	vector<EssentialBC>
	readEssentialBoundaryConditionsFromFile(string essentialBCs) {
		string line_bc;
		string partial;
		istringstream iss;

		ifstream file_bc(essentialBCs.c_str());

		getline(file_bc, line_bc);
		size_t numberOfBCs = atoi(line_bc.c_str());

		size_t nodeId;
		size_t direction;
		double value;
		size_t flag;
		vector<EssentialBC> EssentialBoundaryConditions(numberOfBCs);

		for (unsigned int index = 0; index < numberOfBCs; index++) {
			getline(file_bc, line_bc);
			iss.str(line_bc);
			getline(iss, partial, ' ');
			nodeId = atoi(partial.c_str());
			getline(iss, partial, ' ');
			direction = atoi(partial.c_str());
			getline(iss, partial, ' ');
			value = atof(partial.c_str());
			getline(iss, partial, ' ');
			flag = atoi(partial.c_str());
			iss.clear();
			EssentialBC es = EssentialBC(nodeId, direction, value, flag);
			EssentialBoundaryConditions[index] = es;
		}
		return EssentialBoundaryConditions;
	}


	
//}

#endif  // UTILITIES_H
