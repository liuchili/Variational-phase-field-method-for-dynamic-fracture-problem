// -*- C++ -*-
#ifndef CONJUGATEGRADIENTSOLVER_H
#define CONJUGATEGRADIENTSOLVER_H

#include "Definitions.h"

VectorXd
computeMatrixVectorProductPerTask(const vector<vector<double> > & matrixValuesPerTask, const vector<vector<size_t> > & MatrixIndicesPerTask, 
	const VectorXd & allVectorValues) {
	
	// product size equals the number of rows stored in componentMatrixValues;
	VectorXd OperationProduct(matrixValuesPerTask.size());
	double dotProductPerRow = 0.;
	#pragma omp parallel for schedule(dynamic) private(dotProductPerRow)
	for (size_t rowIndex = 0; rowIndex < matrixValuesPerTask.size(); rowIndex++) {
		dotProductPerRow = 0.;
		for (size_t localColumnIndex = 0; localColumnIndex < matrixValuesPerTask[rowIndex].size(); localColumnIndex++) {
			dotProductPerRow += matrixValuesPerTask[rowIndex][localColumnIndex]*allVectorValues(MatrixIndicesPerTask[rowIndex][localColumnIndex]);
		}
		OperationProduct(rowIndex) = dotProductPerRow;
	}
	return OperationProduct;
}


double
computeTwoVectorInnerProductPerTask(const VectorXd & firstVectorValuesPerTask, const VectorXd & secondVectorValuesPerTask) {

	double result = 0.;
	#pragma omp parallel for reduction(+:result)
	for (size_t index = 0; index < firstVectorValuesPerTask.size(); index++) {
		result += firstVectorValuesPerTask(index)*secondVectorValuesPerTask(index);
	}
	return result;
}


/*
VectorXd
computeDiagonalMatrixVectorProductPerTask(const VectorXd & diagonalMatrixValuesPerTask, const VectorXd & vectorValuesPerTask) {
	const size_t numberOfRows = diagonalMatrixValuesPerTask.size();
	VectorXd result(numberOfRows);
	#pragma omp parallel for schedule(static)
	for (size_t index = 0; index < numberOfRows; index++) {
		result(index) = diagonalMatrixValuesPerTask(index)*vectorValuesPerTask(index);
	}
	return result;
}
*/
void
computeTwoVectorProductElementWise(const VectorXd & firstVector, const VectorXd & secondVector,
	VectorXd & resultVector) {
	#pragma omp parallel for schedule (static)
	for (size_t index = 0; index < firstVector.size(); index++) {
		resultVector(index) = firstVector(index)*secondVector(index);
	}
}

void
updateAVectorByAnotherPerTask(VectorXd & firstVectorValuesPerTask, const VectorXd & secondVectorValuesPerTask, 
	const double firstCoefficient, const double secondCoefficient) {

	size_t vectorSize = firstVectorValuesPerTask.size();
	#pragma omp parallel for schedule(static)
	for (size_t index = 0; index < vectorSize; index++) {
		firstVectorValuesPerTask(index) = firstCoefficient*firstVectorValuesPerTask(index)+secondCoefficient*secondVectorValuesPerTask(index);
	}
}


/*
vector<double>
computeForwardSubstitution(const vector<vector<double> > & allLowerTriangularMatrixValues, const vector<vector<size_t> > & allLowerTriangularMatrixIndices,
	const vector<double> & allVectorValues) {

    size_t totalRowNumber = allLowerTriangularMatrixValues.size();
	vector<double> result(totalRowNumber);
	result[0] = allVectorValues[0]/allLowerTriangularMatrixValues[0][0];
	double updateResult = 0.;
	size_t subRowSize = 0;
	for (size_t rowIndex = 1; rowIndex < totalRowNumber; rowIndex++) {
		result[rowIndex] = 0.;
		updateResult = allVectorValues[rowIndex];
		subRowSize = allLowerTriangularMatrixValues[rowIndex].size();
		for (size_t localColumnIndex = 0; localColumnIndex < subRowSize-1; localColumnIndex++) {
			updateResult -= allLowerTriangularMatrixValues[rowIndex][localColumnIndex]*result[allLowerTriangularMatrixIndices[rowIndex][localColumnIndex]];
		}
		updateResult /= allLowerTriangularMatrixValues[rowIndex][subRowSize-1];
		result[rowIndex] = updateResult;
	}
	return result;
}

vector<double>
computeBackwardSubsitution(const vector<vector<double> > & allUpperTriangularMatrixValues, const vector<vector<size_t> > & allUpperTriangularMatrixIndices,
	const vector<double> & allVectorValues) {

	size_t totalRowNumber = allUpperTriangularMatrixValues.size();
	vector<double> result(totalRowNumber);
	result[totalRowNumber-1] = allVectorValues[totalRowNumber-1]/allUpperTriangularMatrixValues[totalRowNumber-1][0];
	double updateResult = 0;
	size_t subRowSize = 0;
	size_t actualRowIndex = 0;
	for (size_t rowIndex = 1; rowIndex < totalRowNumber; rowIndex++) {
		actualRowIndex = totalRowNumber-1-rowIndex;
		result[actualRowIndex] = 0.;
		updateResult = allVectorValues[actualRowIndex];
		subRowSize = allUpperTriangularMatrixValues[actualRowIndex].size();
		for (size_t localColumnIndex = 1; localColumnIndex < subRowSize; localColumnIndex++) {
			updateResult -= allUpperTriangularMatrixValues[actualRowIndex][localColumnIndex]*result[allUpperTriangularMatrixIndices[actualRowIndex][localColumnIndex]];
		}
		updateResult /= allUpperTriangularMatrixValues[actualRowIndex][0];
		result[actualRowIndex] = updateResult;
	}
	return result;
}
*/

VectorXd
extractInverseDiagonalEntriesAndContainerFormPerTask(const SparseMatrix<double, RowMajor>  & matAll, const int startingRowIndexPerTask, const int numberOfRowsPerTask,
	vector<vector<double> > & matrixValuesPerTask, vector<vector<size_t> > & matrixIndicesPerTask) {


	matrixValuesPerTask.resize(numberOfRowsPerTask);
	matrixIndicesPerTask.resize(numberOfRowsPerTask);
	VectorXd diagonalInversePerTask(numberOfRowsPerTask);
	size_t localRowIndex = 0;

	#pragma omp parallel for schedule(dynamic) private(localRowIndex)
	for (size_t rowIndex = startingRowIndexPerTask; rowIndex < startingRowIndexPerTask+numberOfRowsPerTask; ++rowIndex) {
		localRowIndex = rowIndex - startingRowIndexPerTask;
		for (SparseMatrix<double, RowMajor>::InnerIterator it(matAll,rowIndex); it; ++it) {
			if (it.col()==rowIndex) {
				diagonalInversePerTask(localRowIndex) = 1./it.valueRef();
			}
			matrixValuesPerTask[localRowIndex].push_back(it.valueRef());
			matrixIndicesPerTask[localRowIndex].push_back(it.col());
		}
	}
	return diagonalInversePerTask;
}


VectorXd
computeSolutionUsingConjugateGradient (const SparseMatrix<double, RowMajor> & matPerTask, const VectorXd & bValuesAll, const VectorXd & diagonalEntries,
	const vector<int> & startingRowIndexForEachTask, const vector<int> & numberOfRowForEachTask, const double tolerance, 
	size_t & numIter, double & residue) {

    // taking initial guess to always be zero
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int startingRowIndices[startingRowIndexForEachTask.size()];
    int numberOfRows[startingRowIndexForEachTask.size()];
    copy(startingRowIndexForEachTask.begin(), startingRowIndexForEachTask.end(), startingRowIndices);
    copy(numberOfRowForEachTask.begin(), numberOfRowForEachTask.end(), numberOfRows);

    const size_t            totalRowNumber = bValuesAll.size();

    vector<vector<double> > matrixValuesPerTask(numberOfRowForEachTask[rank]);
    vector<vector<size_t> > matrixIndicesPerTask(numberOfRowForEachTask[rank]);

    VectorXd                diagonalInversePerTask(numberOfRowForEachTask[rank]);
    VectorXd                residueCurrentPerTask(numberOfRowForEachTask[rank]);
    VectorXd                pVectorPerTask(numberOfRowForEachTask[rank]);
    VectorXd                zVectorPerTask(numberOfRowForEachTask[rank]);
    VectorXd                solutionPerTask(numberOfRowForEachTask[rank]);


    //VectorXd                pMatrixProductPerTask(numberOfRowForEachTask[rank]);
    VectorXd                pMatrixProductPerTask(totalRowNumber);


    VectorXd                pVectorAll(totalRowNumber);
    VectorXd                solutionAll(totalRowNumber);
    VectorXd                pMatrixProductAll(totalRowNumber);


    double                  alpha = 0.;
    double                  beta = 0.;
    double                  r_z_perTask = 0.;
    double                  r_z_perTask_New = 0.;
    double                  r_z_all = 0.;
    double                  r_z_all_New = 0.;
    double                  p_A_p_perTask = 0.;
    double                  p_A_p_all = 0.;
    double                  r_square_perTask = 0.;
    double                  r_square_all = 0.;
    size_t                  localRowIndex = 0;
    const size_t            maxIteration = totalRowNumber;
                            numIter = 100;
                            residue = 1.;

    // initialization-------------------------------------------------------------------------------------------------------------------------------
    //diagonalInversePerTask = extractInverseDiagonalEntriesAndContainerFormPerTask(matAll, startingRowIndexForEachTask[rank], 
    //	numberOfRowForEachTask[rank], matrixValuesPerTask, matrixIndicesPerTask);

    #pragma omp parallel for schedule(static) private(localRowIndex)
    for (size_t rowIndex = startingRowIndexForEachTask[rank]; rowIndex < startingRowIndexForEachTask[rank]+numberOfRowForEachTask[rank]; rowIndex++) {
    	localRowIndex = rowIndex - startingRowIndexForEachTask[rank];
    	residueCurrentPerTask(localRowIndex) = bValuesAll(rowIndex);
    	solutionPerTask(localRowIndex) = 0.;
    	diagonalInversePerTask(localRowIndex) = 1./diagonalEntries(rowIndex);
    }
    //zVectorPerTask = computeDiagonalMatrixVectorProductPerTask(diagonalInversePerTask, residueCurrentPerTask);
    computeTwoVectorProductElementWise(diagonalInversePerTask, residueCurrentPerTask, zVectorPerTask);

    pVectorPerTask = zVectorPerTask;
    MPI_Allgatherv(pVectorPerTask.data(), numberOfRows[rank], MPI_DOUBLE, pVectorAll.data(), numberOfRows, startingRowIndices, MPI_DOUBLE, MPI_COMM_WORLD);

    // iteration--------------------------------------------------------------------------------------------------------------------------------------
    for (size_t ii = 0; ii < maxIteration; ii++) {


    	//r_z_perTask = computeTwoVectorInnerProductPerTask(residueCurrentPerTask, zVectorPerTask);
        r_z_perTask = residueCurrentPerTask.dot(zVectorPerTask);

    	//pMatrixProductPerTask = computeMatrixVectorProductPerTask(matrixValuesPerTask, matrixIndicesPerTask, pVectorAll);
        pMatrixProductPerTask = matPerTask*pVectorAll;

    	//p_A_p_perTask = computeTwoVectorInnerProductPerTask(pVectorPerTask, pMatrixProductPerTask);
    	p_A_p_perTask = pVectorAll.dot(pMatrixProductPerTask);

    	MPI_Allreduce(&r_z_perTask, &r_z_all, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    	MPI_Allreduce(&p_A_p_perTask, &p_A_p_all, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    	MPI_Allreduce(pMatrixProductPerTask.data(), pMatrixProductAll.data(), totalRowNumber, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    	alpha = r_z_all/p_A_p_all;

    	//updateAVectorByAnotherPerTask(solutionPerTask, pVectorPerTask, 1, alpha);
    	//updateAVectorByAnotherPerTask(residueCurrentPerTask, pMatrixProductPerTask, 1, -1.*alpha);
    	solutionPerTask += alpha*pVectorPerTask;
    	residueCurrentPerTask -= alpha*pMatrixProductAll.block(startingRowIndexForEachTask[rank], 0, numberOfRowForEachTask[rank], 1);

    	//r_square_perTask = computeTwoVectorInnerProductPerTask(residueCurrentPerTask, residueCurrentPerTask);
    	r_square_perTask = residueCurrentPerTask.dot(residueCurrentPerTask);
    	MPI_Allreduce(&r_square_perTask, &r_square_all, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    	residue = sqrt(r_square_all);

    	if (residue < tolerance) {
    		numIter = ii;
    		break;
    	}

    	//zVectorPerTask = computeDiagonalMatrixVectorProductPerTask(diagonalInversePerTask, residueCurrentPerTask);
    	computeTwoVectorProductElementWise(diagonalInversePerTask, residueCurrentPerTask, zVectorPerTask);
    	//r_z_perTask_New = computeTwoVectorInnerProductPerTask(residueCurrentPerTask, zVectorPerTask);
    	r_z_perTask_New = residueCurrentPerTask.dot(zVectorPerTask);

    	MPI_Allreduce(&r_z_perTask_New, &r_z_all_New, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    	beta = r_z_all_New/r_z_all;

    	//updateAVectorByAnotherPerTask(pVectorPerTask, zVectorPerTask, beta, 1.);
        pVectorPerTask = beta*pVectorPerTask+zVectorPerTask;
    	MPI_Allgatherv(pVectorPerTask.data(), numberOfRows[rank], MPI_DOUBLE, pVectorAll.data(), numberOfRows, startingRowIndices, MPI_DOUBLE, MPI_COMM_WORLD);
    }

    MPI_Allgatherv(solutionPerTask.data(), numberOfRows[rank], MPI_DOUBLE, solutionAll.data(), numberOfRows, startingRowIndices, MPI_DOUBLE, MPI_COMM_WORLD);

    return solutionAll;
}

#endif  // ESSENTIALBOUNDARYCONDITIONS_H
