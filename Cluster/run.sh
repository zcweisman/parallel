mpirun -mca plm_rsh_no_tree_spawn 1 -x LD_LIBRARY_PATH -np 4 -H master,slave1,slave2,slave3 /home/ubuntu/Cluster/bin/application
