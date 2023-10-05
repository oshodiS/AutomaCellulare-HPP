## Questo e' un frammento del Makefile utilizzato negli esempi
## illustrati durante il corso. Questo file puo' essere modificato e
## personalizzato in base alle proprie esigenze. Si puo' anche
## decidere di non usarlo; in tal caso indicare le istruzioni di
## compilazione nel file README presente nella directory a livello
## superiore.
##
## Se si decide di usare questo makefile, il comando "make" dovrebbe
## compilare tutti i programmi consegnati senza errori né warning.  Si
## consiglia pertanto di rimuovere eventuali target non presenti
## nell'archivio consegnato.
##
## Questo Makefile compila i file "omp-*.c" usando il flag -fopenmp, i
## file "cuda-*.cu" con il compilatore nvcc, e i file "mpi-*.c" con
## mpicc.
##
## I principali target definiti da questo makefile sono:
##
## make         compila tutti i sorgenti disponibili
## make clean   cancella i file temporanei e gli eseguibili
## make openmp  compila la versione OpenMP
## make mpi     compila la versione MPI

EXE_OMP:=$(basename $(wildcard omp-*.c))
EXE_MPI:=$(basename $(wildcard mpi-*.c))
DATAFILES:=
EXE_SERIAL:=hpp
EXE:=$(EXE_OMP) $(EXE_MPI) $(EXE_SERIAL) $(EXE_CUDA)
CFLAGS+=-std=c99 -Wall -Wpedantic -O2
LDLIBS+=-lm
NVCC:=nvcc
NVCFLAGS+=
NVLDLIBS+=-lm

.PHONY: clean

ALL: $(EXE)

movie:
	$(CC) $(CFLAGS) -DDUMP_ALL hpp.c -o hpp-movie $(LDLIBS)
	./hpp-movie 256 256 cannon.in
	ffmpeg -y -i "hpp%05d.pgm" -vcodec mpeg4 movie.avi
movie-mpi:
	mpicc  -DDUMP_ALL mpi-hpp.c -o mpi-hpp-movie $(LDLIBS)
	mpirun mpi-hpp-movie 256 256 cannon.in
	ffmpeg -y -i "mpi-hpp%05d.pgm" -vcodec mpeg4 movie.avi
movie-omp:
	gcc -DDUMP_ALL -fopenmp  omp-hpp.c -o omp-hpp-movie $(LDLIBS)
	./omp-hpp-movie 256 256 cannon.in
	ffmpeg -y -i "omp-hpp%05d.pgm" -vcodec mpeg4 movie.avi


$(EXE_OMP): CFLAGS+=-fopenmp
openmp: $(EXE_OMP)

$(EXE_MPI): CC=mpicc
mpi: $(EXE_MPI)

clean:
	\rm -f $(EXE) hpp-movie *.o *~ *.pbm *.pgm *.avi

