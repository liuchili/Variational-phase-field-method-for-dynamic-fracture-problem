/*
 * main.cpp
 *
 *  Created on: August 4, 2014
 *      Author: Reid
 */
 

#include "ReadInputFiles.h"
#include "NewtonRaphson.h"

int main(int argc, char *argv[]) {

    int provided;
    int rank;
    int size;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    if (rank == 0) {
        cout << "Running MainTest" << endl;
        cout << "using " << size << " tasks with " << 
        omp_get_max_threads()  << " threads_per_task for a total of " << 
        size*omp_get_max_threads() << " threads." << endl;
        cout << "Eigen uses threads " << Eigen::nbThreads() << endl;       
    } 
    
    char   tempfname[100];
    char   sr[100];
    char   sr1[100];
    sprintf(sr,"PFTests");
    sprintf(sr1,"Kalthoff");
    size_t elementId = 39837;
    //size_t elementId = 22548;
    int outstep = 1;

    int casee    =  atoi(argv[1]);
    double k     =  atof(argv[2]);
    size_t niter =  atoi(argv[3]);
    size_t nstep =  atoi(argv[4]);
    double dt    =  atof(argv[5]);

    double V0           = 16500.;
    double t0           = 1e-6;
    double dampingAlpha = 0.;
    double dampingBeta  = 0.;
    double beta         = 0.25;
    double gamma        = 0.5;

    sprintf(tempfname, "/scratch16/kramesh1/liuchili/%s/%s/initialNodePositionsPhis.dat",sr, sr1);
    string filenameone = tempfname;

    sprintf(tempfname, "/scratch16/kramesh1/liuchili/%s/%s/initialNodeVelocities.dat",sr, sr1);
    string filenametwo = tempfname;

    sprintf(tempfname, "/scratch16/kramesh1/liuchili/%s/%s/initialNodeAccelerations.dat",sr, sr1);
    string filenamethree = tempfname;

    sprintf(tempfname, "/scratch16/kramesh1/liuchili/%s/%s/nodeMeshes.dat", sr, sr1);
    string filenamefour = tempfname;

    sprintf(tempfname, "/scratch16/kramesh1/liuchili/%s/%s/essentialBoundaryConditions.dat", sr, sr1);
    string filenamefive = tempfname;


    vector<EssentialBC> essentialBCs = readEssentialBoundaryConditionsFromFile(filenamefive);
    WorldDynamic world = generateWorldFromFile(filenameone, filenametwo, filenamethree, filenamefour, k, size);

    if (rank == 0) {
        cout << "world has number of nodes " << world.getWorldNumberOfNodes() << endl;
        cout << "world has number of elements " << world.getWorldNumberOfElements() << endl;
        cout << "essentialBCs size is " << essentialBCs.size() << endl;
        cout << "l0 and k is " << world.getWorldElements()[0].getMaterialModels()[0]->getL0() << " " << world.getK() << endl;
        cout << "Gc and dt is " << world.getWorldElements()[0].getMaterialModels()[0]->getToughness() << " " << dt << endl;
        cout << "density is " << world.getWorldElements()[0].getMaterialModels()[0]->getDensity() << " " << endl;
    }
    
   Vector2d acceleration;
   acceleration.fill(0.);

    vector<size_t>     elementChunkSize                 =  world.getElementChunkSize();
    vector<size_t>     elementOffsetSize                =  world.getElementOffsetSize();
    vector<int>     phiStartingIndex                    =  world.getPhiStartingRowIndices();
    vector<int>     phiNumberOfRow                      =  world.getPhiNumberOfRows();
    //vector<int>     nodeDisplacementStartingIndex    =  world.getNodeBasedDisplacementStartingIndex();
    //vector<int>     nodeDisplacementChunkSize        =  world.getNodeBasedDisplacementChunkSize();

    int    rank_output = 0;
    size_t globalLastElementId = 0;
    size_t localElementId = 0;
    for (size_t ii = 0; ii < elementChunkSize.size(); ii++) {
        globalLastElementId += elementChunkSize[ii]-1;
        if (globalLastElementId > elementId) {
            rank_output = ii;
            localElementId = elementId - elementOffsetSize[ii];
            break;
        }
    }

    size_t idcheck;
    if (rank_output == 0) {
        idcheck = localElementId;
    }
    else {
        idcheck = localElementId+elementOffsetSize[rank_output];
    }
    if (rank == 0) {
        cout << "checking back id " << idcheck << endl;
        cout << "out put rank is " << rank_output << endl;
    }


    char PositionPhiFile[100];
    char VelocityFile[100];
    char AccelerationFile[100];
    char NodeStressFile[100];
    char NodeStrainFile[100];
    char DisplacementFile[100];
    char TimeEnergyFile[100];

    const size_t totalNumberOfNodes = world.getWorldNumberOfNodes();

    FILE * PositionPhi;
    FILE * Velocity;
    FILE * Acceleration;
    FILE * NodeStress;
    FILE * NodeStrain;
    FILE * Displacement;
    FILE * TimeEnergy;

    if (rank == rank_output) {

        sprintf(NodeStressFile,"/scratch16/kramesh1/liuchili/%s/%s/%dNodeStresses.dat",sr,sr1,casee);
        sprintf(NodeStrainFile,"/scratch16/kramesh1/liuchili/%s/%s/%dNodeStrains.dat",sr,sr1,casee);
        sprintf(DisplacementFile,"/scratch16/kramesh1/liuchili/%s/%s/%dNodeDisplacements.dat",sr,sr1,casee);
        sprintf(TimeEnergyFile, "/scratch16/kramesh1/liuchili/%s/%s/%dTimeAndEnergy.dat",sr,sr1,casee);

        NodeStress = fopen(NodeStressFile,"w");
        NodeStrain = fopen(NodeStrainFile,"w");
        Displacement = fopen(DisplacementFile,"w");
        TimeEnergy = fopen(TimeEnergyFile, "w");
    }

    sprintf(PositionPhiFile,"/scratch16/kramesh1/liuchili/%s/%s/%dNodePositionsPhis%d.dat",sr, sr1, casee, rank);
    sprintf(VelocityFile, "/scratch16/kramesh1/liuchili/%s/%s/%dNodeVelocities%d.dat",sr, sr1, casee, rank);
    sprintf(AccelerationFile, "/scratch16/kramesh1/liuchili/%s/%s/%dNodeAccelerations%d.dat",sr,sr1,casee, rank);

    PositionPhi = fopen(PositionPhiFile,"w");
    Velocity = fopen(VelocityFile,"w");
    Acceleration = fopen(AccelerationFile,"w");

    VectorXd worldDisplacementVector(world.getWorldNumberOfNodes()*2);
    worldDisplacementVector.fill(0.);
    VectorXd worldPhiVector(world.getWorldNumberOfNodes());
    worldPhiVector.fill(0.);


    vector<Vector2d> worldDisplacementIncrement
      = Utilities::distributeGlobalVectorsToLocalVectors(worldDisplacementVector);
    vector<Vector2d> worldDisplacementAccumulate
      = Utilities::distributeGlobalVectorsToLocalVectors(worldDisplacementVector);

    vector<Vector2d> worldNodalForce
      = Utilities::distributeGlobalVectorsToLocalVectors(worldDisplacementVector);

    vector<double> worldPhiIncrement
      = Utilities::distributeGlobalVectorsToLocalSacalars(worldPhiVector);
    vector<double> worldPhiAccumulate
      = Utilities:: distributeGlobalVectorsToLocalSacalars(worldPhiVector);
    vector<double> totalWorldPhi = worldPhiIncrement;


    vector<double>   worldNodeStressFirst(totalNumberOfNodes);
    vector<double>   worldNodeStressSecond(totalNumberOfNodes);
    vector<double>   worldNodeStressThird(totalNumberOfNodes);

    vector<double>   worldNodeStrainFirst(totalNumberOfNodes);
    vector<double>   worldNodeStrainSecond(totalNumberOfNodes);
    vector<double>   worldNodeStrainThird(totalNumberOfNodes);

    vector<vector<Vector3d> > worldStrainsAtGaussianPointsAccumulate = world.computeWorldStrainsAtGaussianPoints(worldDisplacementIncrement);
    vector<vector<Vector3d> > worldStrainsAtGaussianPointsIncrement = worldStrainsAtGaussianPointsAccumulate;
    vector<vector<double> >   worldPhisAtGaussianPointsAccumulate = world.computeWorldPhisAtGaussianPoints(worldPhiIncrement);
    vector<vector<double> >   worldPhisAtGaussianPointsIncrement = worldPhisAtGaussianPointsAccumulate;

    VectorXd externalForceVector = worldDisplacementVector;


    VectorXd worldDisplacementBodyForceVectorPerTask = world.assembleBodyForceVectorPerTask(acceleration);

    SparseMatrix<double, RowMajor> worldMassMatrixPerTask(totalNumberOfNodes*2, totalNumberOfNodes*2);
    VectorXd worldMassMatrixDiagonalEntriesPerTask = world.assembleMassMatrixAndDiagonalEntriesPerTask(worldMassMatrixPerTask);


    size_t numberIteration = 0;
    size_t updateFlag = 0;
    double delta = 0.;
    Vector2d drivingForce(0.,0.);
    double displacementRelativeError = 0.;
    double phiRelativeError = 0.;

    size_t  iterationDisplacementFinal = 0;
    size_t  iterationPhiFinal = 0;
    double  residueDisplacementFinal = 0.;
    double  residuePhiFinal = 0.;
    double  displacementForceResidue = 0.;
    double  phiForceResidue = 0.;
    double  displacementAbsoluteError = 0.;
    double  phiAbsoluteError = 0.;


    vector<Vector2d> allNodePositions(totalNumberOfNodes);
    vector<Vector2d> allNodeVelocities(totalNumberOfNodes);
    vector<Vector2d> allNodeAccelerations(totalNumberOfNodes);
    vector<double>   allNodePhis(totalNumberOfNodes);

    Matrix<double, 3, 3> tangentMatrix; tangentMatrix.fill(0.);
    vector<Vector2d> eigenValuesVectors(3);
    double incrementX = 0.;
    double velocity = 0.;

    double strainEnergy = 0.;
    double fractureEnergy = 0.;


    double start = omp_get_wtime();

    for (int step = 0; step < nstep; step++) {
        
        if ((step+1)*dt <= t0) {
            incrementX = ((step*dt/t0)*V0+((step+1)*dt/t0)*V0)*dt*0.5;
            velocity = (step+1)*dt/t0*V0;
            if ((step+1)*dt == t0 && rank == 0) {
                cout <<  "At step " << step+1 << " impactor reaches the target velocity " << V0 << " mm/s" << endl;
            }
        }
        else {
            incrementX = V0*dt;
            velocity = V0;
        }


        #pragma omp parallel for schedule(static)
        for (unsigned int bcindex = 0; bcindex < essentialBCs.size(); bcindex++) {
            if (essentialBCs[bcindex].getFlag() == 1) {
                essentialBCs[bcindex].changeBoundaryValue(incrementX);
            }
        }

        // before each iteration, reinitialize increments to zeros
        worldDisplacementIncrement = Utilities::distributeGlobalVectorsToLocalVectors(worldDisplacementVector);
        worldPhiIncrement = Utilities::distributeGlobalVectorsToLocalSacalars(worldPhiVector);

        // enforcing essential boundary conditions to displacement increments
        #pragma omp parallel for schedule(static)
        for (unsigned int bcIndex = 0; bcIndex < essentialBCs.size(); bcIndex++) {
            worldDisplacementIncrement[essentialBCs[bcIndex].getNodeId()]
              (essentialBCs[bcIndex].getDirection()) = essentialBCs[bcIndex].getValue();
        }


        auto start1 = high_resolution_clock::now();
        solveDisplacementAndPhiFieldThroughNewtonRaphson(world, niter, worldDisplacementIncrement, worldPhiIncrement, 
            numberIteration, worldStrainsAtGaussianPointsAccumulate, 
            worldPhisAtGaussianPointsAccumulate, worldDisplacementAccumulate, worldPhiAccumulate,
            essentialBCs, externalForceVector, dampingAlpha, dampingBeta, gamma, beta, dt,
            worldDisplacementBodyForceVectorPerTask, worldMassMatrixDiagonalEntriesPerTask, worldMassMatrixPerTask, 
            displacementRelativeError, phiRelativeError, displacementAbsoluteError, phiAbsoluteError,
            iterationDisplacementFinal, iterationPhiFinal, residueDisplacementFinal, residuePhiFinal, displacementForceResidue, phiForceResidue);
        auto stop1 = high_resolution_clock::now();
        auto duration1 = duration_cast<microseconds>(stop1 - start1);


        worldStrainsAtGaussianPointsIncrement = world.computeWorldStrainsAtGaussianPoints(worldDisplacementIncrement);
        worldPhisAtGaussianPointsIncrement = world.computeWorldPhisAtGaussianPoints(worldPhiIncrement);

        // now update quantity accumulate use increment
        #pragma omp parallel for schedule (static)
        for (size_t nindex = 0; nindex < world.getWorldNumberOfNodes(); nindex++) {
            worldDisplacementAccumulate[nindex] += worldDisplacementIncrement[nindex];
            worldPhiAccumulate[nindex] += worldPhiIncrement[nindex];
        }

        #pragma omp parallel for schedule (static)
        for (size_t eindex = elementOffsetSize[rank]; eindex < elementOffsetSize[rank]+elementChunkSize[rank]; eindex++) {
            for (unsigned int qpindex = 0; qpindex < worldStrainsAtGaussianPointsIncrement[eindex].size(); qpindex++) {
                worldStrainsAtGaussianPointsAccumulate[eindex][qpindex] += worldStrainsAtGaussianPointsIncrement[eindex][qpindex];
                worldPhisAtGaussianPointsAccumulate[eindex][qpindex] += worldPhisAtGaussianPointsIncrement[eindex][qpindex];
            }
        }

        // update position (geometrical matrices) PER RANK
        world.updatWorldNodePositionsVelocitiesAccelerations(worldDisplacementIncrement, beta, gamma, dt);
        //world.updateWorldNodePhis(worldPhiIncrement);


        if (step%outstep == 0) {
            //worldNodalForce = Utilities::distributeGlobalVectorsToLocalVectors(world.assembleForceVector(worldDisplacementAccumulate));
            world.computeWorldNodalStresses(worldDisplacementAccumulate, worldNodeStressFirst, worldNodeStressSecond, worldNodeStressThird);
            world.computeWorldNodalStrains(worldDisplacementAccumulate, worldNodeStrainFirst, worldNodeStrainSecond, worldNodeStrainThird);
            strainEnergy = world.assembleStrainEnergy(worldStrainsAtGaussianPointsAccumulate);
            fractureEnergy = world.assembleFractureEnergy(worldPhiAccumulate, worldPhisAtGaussianPointsAccumulate);
            allNodePositions = world.getWorldNodePositions();
            allNodeVelocities = world.getWorldNodeVelocities();
            allNodeAccelerations = world.getWorldNodeAccelerations();

            for (size_t nindex = phiStartingIndex[rank]; nindex < phiStartingIndex[rank]+phiNumberOfRow[rank]; nindex++) {
                fprintf(PositionPhi, "%d %4.7f %4.7f %4.7f\n", nindex, allNodePositions[nindex](0), allNodePositions[nindex](1), worldPhiAccumulate[nindex]);
                fprintf(Velocity, "%d %4.7f %4.7f\n", nindex, allNodeVelocities[nindex](0), allNodeVelocities[nindex](1));
                fprintf(Acceleration, "%d %4.7f %4.7f\n", nindex, allNodeAccelerations[nindex](0), allNodeAccelerations[nindex](1));
            }
            if (rank == rank_output) {

                tangentMatrix = world.getOneElement(elementId).getMaterialModels()[0]->getTangentMatrix();

                
                for (unsigned int nindex = 0; nindex < totalNumberOfNodes; nindex++) {
                    fprintf(NodeStress, "%d %4.7f %4.7f %4.7f\n", nindex, worldNodeStressFirst[nindex], 
                        worldNodeStressSecond[nindex], worldNodeStressThird[nindex]);

                    fprintf(NodeStrain, "%d %4.7f %4.7f %4.7f\n", nindex, worldNodeStrainFirst[nindex], 
                        worldNodeStrainSecond[nindex], worldNodeStrainThird[nindex]);

                    fprintf(Displacement, "%d %4.6f %4.6f\n", nindex, worldDisplacementAccumulate[nindex](0), 
                        worldDisplacementAccumulate[nindex](1));
                }
                fprintf(TimeEnergy, "%4.8f %4.6f %4.6f\n", dt*(step+1), strainEnergy, fractureEnergy);
                
                cout << "Time solving for step " << step+1 <<  " with iteration " << numberIteration << " is " 
                << duration1.count()/1000000. << " seconds and, absolute, relative errors and force residues are " << displacementAbsoluteError << " " << phiAbsoluteError << " " << displacementRelativeError << " " << phiRelativeError << " " <<
                displacementForceResidue << " " << phiForceResidue << endl;
                cout << "At this step at the last iteration the CG solver uses iterations with residue error " << iterationDisplacementFinal << " " << iterationPhiFinal << " " << 
                residueDisplacementFinal << " " << residuePhiFinal << endl;
                cout << elementId << "th element has tangent matrix " << endl;
                cout << tangentMatrix << endl;

                cout << elementId << "th element has phi: " << worldPhisAtGaussianPointsAccumulate[elementId][0] << " " << worldPhisAtGaussianPointsAccumulate[elementId][1] << " " <<
                worldPhisAtGaussianPointsAccumulate[elementId][2] << endl;            

                cout << endl;
            }
        } 
    }
    fclose(PositionPhi);
    fclose(Velocity);
    fclose(Acceleration);

    if (rank == rank_output) {
        fclose(NodeStress);
        fclose(NodeStrain);
        fclose(Displacement);
        fclose(TimeEnergy);
        double duration = omp_get_wtime() - start;
        printf("Time taken: %dh, %dm, %2.2fs\n", int(floor(duration/3600.)), 
        -int(floor(duration/3600.))+int(floor(fmod(duration/60., 60.))), 
        -floor(fmod(duration/60., 60.))*60. + fmod(duration, 3600.) );
        cout << "program terminated" << endl;
    }
    MPI_Finalize();
    return 0;
}