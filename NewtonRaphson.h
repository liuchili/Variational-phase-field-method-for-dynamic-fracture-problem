// -*- C++ -*-
#ifndef NEWTONRAPHSON_H
#define NEWTONRAPHSON_H

#include "Utilities_Liuchi.h"
#include "ConjugateGradientSolver.h"


void
solveDisplacementAndPhiFieldThroughNewtonRaphson(
	WorldDynamic & world, const size_t & niter,
	vector<Vector2d> & worldDisplacementIncrement, // essential boundary conditions enforced, otherwise all zero
	vector<double>   & worldPhiIncrement, // always all zero as input
	size_t & iteration,
	const vector< vector<Vector3d> > & worldStrainsAtGaussianPointsAccumulate,
	const vector< vector<double> >   & worldPhisAtGaussianPointsAccumulate,
	const vector<Vector2d> & worldDisplacementAccumulate,
	const vector<double>   & worldPhiAccumulate,
	const vector<EssentialBC>  & essentialBoundaryConditions,
	const VectorXd  & externalForceVector,
	//const Vector2d & acceleration,
	const double & dampingAlpha,
	const double & dampingBeta,
	const double & gamma,
	const double & beta,
	const double & dt,
    const VectorXd & worldDisplacementBodyForceVectorPerTask,
    const VectorXd & worldMassMatrixDiagonalEntriesPerTask,
    const SparseMatrix<double, RowMajor> & worldMassMatrixPerTask,
	double & displacementRelativeError,
	double & phiRelativeError,
	double & displacementAbsoluteError,
	double & phiAbsoluteError,
	size_t & iterationDisplacementFinal,
	size_t & iterationPhiFinal,
	double & residueDisplacementFinal,
	double & residuePhiFinal,
	double & DisplacementForceResidue,
	double & PhiForceResidue) {

	int rank = 0;
	int size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); 
    MPI_Comm_size(MPI_COMM_WORLD, &size);

	const size_t nDofDisplacement = world.getWorldNumberOfNodes()*2;
	const size_t nDofPhi = world.getWorldNumberOfNodes();
	size_t updateFlag = 1;

	const vector<int>   displacementStartingRowIndices = world.getDisplacementStartingRowIndices();
	const vector<int>   displacementNumberOfRows =       world.getDisplacementNumberOfRows();
	const vector<int>   phiStartingRowIndices =          world.getPhiStartingRowIndices();
	const vector<int>   phiNumberOfRows =                world.getPhiNumberOfRows();


	iterationDisplacementFinal = 0;
	iterationPhiFinal = 0;
	residueDisplacementFinal = 0.;
	residuePhiFinal = 0.;
	DisplacementForceResidue = 1.;
	PhiForceResidue = 1.;

	displacementRelativeError = 1.;
	phiRelativeError = 1.;
	displacementAbsoluteError = 0.;
	phiAbsoluteError = 0.;


	VectorXd worldDisplacementIncrementVectorCurrent = 
	   Utilities::convertLocalNodalDisplacementToGlobalList(worldDisplacementIncrement); 
	VectorXd worldPhiIncrementVectorCurrent = 
	   Utilities::convertLocalNodalPhiToGlobalList(worldPhiIncrement);

	VectorXd worldDisplacementAccumulateVector = Utilities::convertLocalNodalDisplacementToGlobalList(worldDisplacementAccumulate);
	VectorXd worldPhiAccumulateVector = Utilities::convertLocalNodalPhiToGlobalList(worldPhiAccumulate);

    VectorXd worldDisplacementStiffnessMatrixDiagonalEntriesPerTask(nDofDisplacement); worldDisplacementStiffnessMatrixDiagonalEntriesPerTask.fill(0.);
    //VectorXd worldMassMatrixDiagonalEntriesPerTask(nDofDisplacement); worldMassMatrixDiagonalEntriesPerTask.fill(0.);
    VectorXd worldTangentMatrixDiagonalEntries(nDofDisplacement); worldTangentMatrixDiagonalEntries.fill(0.);
    VectorXd worldPhiStiffnessMatrixDiagonalEntriesPerTask(nDofPhi); worldPhiStiffnessMatrixDiagonalEntriesPerTask.fill(0.);
    VectorXd worldPhiStiffnessMatrixDiagonalEntries(nDofPhi); worldPhiStiffnessMatrixDiagonalEntries.fill(0.);

	
	//VectorXd worldDisplacementBodyForceVectorPerTask = world.assembleBodyForceVectorPerTask(acceleration);
	VectorXd worldBodyPhiForceVectorPerTask(nDofPhi); worldBodyPhiForceVectorPerTask.fill(0.);


	SparseMatrix<double, RowMajor> worldDisplacementStiffnessMatrixPerTask(nDofDisplacement, nDofDisplacement);
	world.assembleStiffnessMatrixAsIsPerTask(worldPhisAtGaussianPointsAccumulate, 
		worldStrainsAtGaussianPointsAccumulate, worldDisplacementStiffnessMatrixPerTask);


   // SparseMatrix<double, RowMajor> worldMassMatrixPerTask(nDofDisplacement, nDofDisplacement);
   // worldMassMatrixDiagonalEntriesPerTask = world.assembleMassMatrixAndDiagonalEntriesPerTask(worldMassMatrixPerTask);


	SparseMatrix<double, RowMajor> worldDampingMatrixPerTask(nDofDisplacement, nDofDisplacement);
	worldDampingMatrixPerTask = dampingAlpha*worldDisplacementStiffnessMatrixPerTask+dampingBeta*worldMassMatrixPerTask;
	worldDampingMatrixPerTask.makeCompressed();


    SparseMatrix<double, RowMajor> worldMixingMatrixPerTask(nDofDisplacement, nDofDisplacement);
	VectorXd inertiaDampingForceVectorPerTask(nDofDisplacement); inertiaDampingForceVectorPerTask.fill(0.);

	VectorXd worldMassKinematicVector(nDofDisplacement);
	VectorXd worldDampingKinematicVector(nDofDisplacement);
	world.getWorldMassKinematicVector(beta, dt, worldMassKinematicVector);
	world.getWorldDampingKinematicVector(beta, gamma, dt, worldDampingKinematicVector);
	VectorXd previousForceVectorPerTask = worldMassMatrixPerTask*worldMassKinematicVector+worldDampingMatrixPerTask*worldDampingKinematicVector;

	SparseMatrix<double, RowMajor> worldPhiStiffnessMatrixPerTask(nDofPhi, nDofPhi);
    
	VectorXd worldDisplacementForceResidueVector(nDofDisplacement); worldDisplacementForceResidueVector.fill(0.);
	VectorXd worldDisplacementForceResidueVectorPerTask(nDofDisplacement); worldDisplacementForceResidueVectorPerTask.fill(0.);
	VectorXd worldPhiForceResidueVector(nDofPhi); worldPhiForceResidueVector.fill(0.);
	VectorXd worldPhiForceResidueVectorPerTask(nDofPhi); worldPhiForceResidueVectorPerTask.fill(0.);

	VectorXd worldDisplacementPrevious(nDofDisplacement); worldDisplacementPrevious.fill(0.);
	VectorXd worldPhiPrevious(nDofPhi); worldPhiPrevious.fill(0.);

	double relativeErrorDisplacement = 1.;
	double relativeErrorPhi = 1.;
	double absoluteErrorDisplacement = 1.;
	double absoluteErrorPhi = 1.;


	size_t globalLocation = 0;
	size_t iter = 0;

	VectorXd displacementIncrementConjugate(nDofDisplacement); displacementIncrementConjugate.fill(0.);
	VectorXd phiIncrementIncrementConjugate(nDofPhi); phiIncrementIncrementConjugate.fill(0.);

	size_t iterationDisplacement = 0;
	size_t iterationPhi = 0;
	double residueDisplacement = 0;
	double residuePhi = 0;
	double tolerance = 1e-25;

	//while (relativeErrorDisplacement > 1e-8 || relativeErrorPhi > 1e-8) {
    while (!((relativeErrorDisplacement+relativeErrorPhi) < 1e-5 && DisplacementForceResidue < 1e-5)) {

    	worldDisplacementPrevious = worldDisplacementIncrementVectorCurrent;
    	worldPhiPrevious = worldPhiIncrementVectorCurrent;

		worldDisplacementStiffnessMatrixDiagonalEntriesPerTask = world.assembleStiffnessMatrixAndDiagonalEntriesPerTask(worldPhiIncrement, worldDisplacementIncrement,
			worldPhisAtGaussianPointsAccumulate, worldStrainsAtGaussianPointsAccumulate, updateFlag, worldDisplacementStiffnessMatrixPerTask);

		worldDampingMatrixPerTask = dampingAlpha*worldDisplacementStiffnessMatrixPerTask+dampingBeta*worldMassMatrixPerTask;
		worldDampingMatrixPerTask.makeCompressed();
		worldMixingMatrixPerTask = 1./beta/dt/dt*worldMassMatrixPerTask+gamma/beta/dt*worldDampingMatrixPerTask;
		worldMixingMatrixPerTask.makeCompressed();


		inertiaDampingForceVectorPerTask = worldMixingMatrixPerTask*worldDisplacementIncrementVectorCurrent;


        worldDisplacementForceResidueVectorPerTask = inertiaDampingForceVectorPerTask;
        worldDisplacementForceResidueVectorPerTask += worldDisplacementStiffnessMatrixPerTask*(worldDisplacementAccumulateVector+worldDisplacementIncrementVectorCurrent);
        worldDisplacementForceResidueVectorPerTask -= worldDisplacementBodyForceVectorPerTask;
        worldDisplacementForceResidueVectorPerTask -= previousForceVectorPerTask;


		//worldDisplacementForceResidueVector = 
		//inertiaDampingForceVector+worldDisplacementStiffnessMatrix*(worldDisplacementAccumulateVector+worldDisplacementIncrementVectorCurrent)-
		//worldDisplacementBodyForceVector-externalForceVector-previousForceVector;


		//worldDisplacementStiffnessMatrixPerTask += 1./beta/dt/dt*worldMassMatrixPerTask+gamma/beta/dt*worldDampingMatrixPerTask;

		worldDisplacementStiffnessMatrixPerTask += worldMixingMatrixPerTask;
		worldDisplacementStiffnessMatrixPerTask.makeCompressed();

        // get the diagonal entries of the global tangent matrix
		worldDisplacementStiffnessMatrixDiagonalEntriesPerTask += dampingAlpha*gamma/beta/dt*worldDisplacementStiffnessMatrixDiagonalEntriesPerTask;
        worldDisplacementStiffnessMatrixDiagonalEntriesPerTask += (1./beta/dt/dt+dampingBeta*gamma/beta/dt)*worldMassMatrixDiagonalEntriesPerTask;


		//before entering CG solver, do MPI all reduce on RHS force vector and on diagonal entry vector
        MPI_Allreduce(worldDisplacementForceResidueVectorPerTask.data(), worldDisplacementForceResidueVector.data(), nDofDisplacement, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(worldDisplacementStiffnessMatrixDiagonalEntriesPerTask.data(), worldTangentMatrixDiagonalEntries.data(), nDofDisplacement, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        worldDisplacementForceResidueVector -= externalForceVector;

        
	    // now impose essential boundary conditions to diagonal entries and force residue vector
	    #pragma omp parallel for schedule(static) private(globalLocation)
		for(unsigned int bcIndex = 0; bcIndex < essentialBoundaryConditions.size(); bcIndex++) {
			globalLocation = essentialBoundaryConditions[bcIndex].getNodeId()*2+essentialBoundaryConditions[bcIndex].getDirection();			
			worldDisplacementForceResidueVector(globalLocation) = 0.;
			worldTangentMatrixDiagonalEntries(globalLocation) = 1.;
			for (SparseMatrix<double, RowMajor>::InnerIterator it(worldDisplacementStiffnessMatrixPerTask, globalLocation); it; ++it) {
				it.valueRef() = 0.;
				if (it.col()==globalLocation) {
					it.valueRef() = 1./size;
				}
			}		
		}
		
        // now do CG computation, taking in essential boundary condition to deal with matrix*vector product
		//displacementIncrementConjugate = computeSolutionUsingConjugateGradient(worldDisplacementStiffnessMatrix, worldDisplacementForceResidueVector,
		//	displacementStartingRowIndices, displacementNumberOfRows, tolerance, iterationDisplacement, residueDisplacement);


        // TO DO BY JACK
        displacementIncrementConjugate = computeSolutionUsingConjugateGradient(worldDisplacementStiffnessMatrixPerTask, worldDisplacementForceResidueVector, 
        	worldTangentMatrixDiagonalEntries, displacementStartingRowIndices, displacementNumberOfRows,
        	tolerance, iterationDisplacement, residueDisplacement);

        
		worldDisplacementIncrementVectorCurrent -= displacementIncrementConjugate;

		worldDisplacementIncrement = Utilities::distributeGlobalVectorsToLocalVectors(worldDisplacementIncrementVectorCurrent);

		//-------------------------------------------------------------------------------------------------------------------------------------------------

		worldBodyPhiForceVectorPerTask = world.assemblePhiBodyForceVectorPerTask(worldDisplacementIncrement, worldStrainsAtGaussianPointsAccumulate);
	
		//phiStiffnessTriplet = world.assemblePhiStiffnessMatrix(worldDisplacementIncrement, worldStrainsAtGaussianPointsAccumulate, updateFlag);
		//worldPhiStiffnessMatrix.setFromTriplets(phiStiffnessTriplet.begin(), phiStiffnessTriplet.end()); 

		worldPhiStiffnessMatrixDiagonalEntriesPerTask = world.assemblePhiStiffnessMatrixAndDiagonalEntriesPerTask(worldDisplacementIncrement, 
			worldStrainsAtGaussianPointsAccumulate, updateFlag, worldPhiStiffnessMatrixPerTask);

		//worldPhiForceResidueVector = worldPhiStiffnessMatrix*(worldPhiAccumulateVector+worldPhiIncrementVectorCurrent)-worldBodyPhiForceVector;
		worldPhiForceResidueVectorPerTask = worldPhiStiffnessMatrixPerTask*(worldPhiAccumulateVector+worldPhiIncrementVectorCurrent);
		worldPhiForceResidueVectorPerTask -= worldBodyPhiForceVectorPerTask;


		// before going into CG, call MPI all reduce to get full diagonal entries and the force residue vector
		MPI_Allreduce(worldPhiForceResidueVectorPerTask.data(), worldPhiForceResidueVector.data(), nDofPhi, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
		MPI_Allreduce(worldPhiStiffnessMatrixDiagonalEntriesPerTask.data(), worldPhiStiffnessMatrixDiagonalEntries.data(), nDofPhi, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

		//phiIncrementIncrementConjugate = computeSolutionUsingConjugateGradient(worldPhiStiffnessMatrix, worldPhiForceResidueVector,
		//	phiStartingRowIndices, phiNumberOfRows, tolerance, iterationPhi, residuePhi);

		phiIncrementIncrementConjugate = computeSolutionUsingConjugateGradient(worldPhiStiffnessMatrixPerTask, worldPhiForceResidueVector, 
			worldPhiStiffnessMatrixDiagonalEntries, phiStartingRowIndices, phiNumberOfRows,
			tolerance, iterationPhi, residuePhi);

		worldPhiIncrementVectorCurrent -= phiIncrementIncrementConjugate;

		worldPhiIncrement = Utilities::distributeGlobalVectorsToLocalSacalars(worldPhiIncrementVectorCurrent);
     
		absoluteErrorDisplacement = (worldDisplacementIncrementVectorCurrent-worldDisplacementPrevious).norm();
		relativeErrorDisplacement = absoluteErrorDisplacement/worldDisplacementPrevious.norm();

		absoluteErrorPhi = (worldPhiIncrementVectorCurrent-worldPhiPrevious).norm();
		relativeErrorPhi = absoluteErrorPhi/worldPhiPrevious.norm();

		DisplacementForceResidue = worldDisplacementForceResidueVector.norm();
		PhiForceResidue = worldPhiForceResidueVector.norm();

		iter++;
		if (iter > niter) {
			break;
		}
	}
	iteration = iter;
	displacementRelativeError = relativeErrorDisplacement;
	phiRelativeError = relativeErrorPhi;
	iterationDisplacementFinal = iterationDisplacement;
	iterationPhiFinal = iterationPhi;
	residueDisplacementFinal = residueDisplacement;
	residuePhiFinal = residuePhi;

	displacementAbsoluteError = absoluteErrorDisplacement;
	phiAbsoluteError = absoluteErrorPhi;
}

#endif // NEWTONRAPHSON_H


