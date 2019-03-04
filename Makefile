OBJS = main.o nn.o readmnist.o

main: ${OBJS}
	g++ -std=c++11 -o run ${OBJS}
main.o: main.cpp
	g++ -std=c++11 -c main.cpp
nn.o: nn.cpp
	g++ -std=c++11 -c nn.cpp
readmnist.o: readmnist.cpp
	g++ -std=c++11 -c readmnist.cpp 

clean:
	rm -f ${OBJS}

