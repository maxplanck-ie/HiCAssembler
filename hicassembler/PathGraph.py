import re


class PathGraph(object):
    """
    This class implements a path graph object.

    """
    def __init__(self):
        """

        Parameters
        ----------

        Returns
        -------

        """
        # initialize the list of contigs as a graph with no edges
        self.node = {}
        self.path = {}
        self.path_id = {}  # maps nodes to paths
        self.adj = {}  # to store the edges

    def __iter__(self):
        """Iterate over the nodes. Use the expression 'for n in S'.

        Returns
        -------
        niter : iterator
            An iterator over all nodes in the graph.

        Examples
        --------
        >>> S = PathGraph()
        >>> S.add_path([0, 1, 2, 3])
        >>> S.add_path([4, 5, 6])
        >>> [x for x in S]
        [0, 1, 2, 3, 4, 5, 6]
        """
        return iter(self.node)

    def __contains__(self, n):
        """Return True if n is a node, False otherwise. Use the expression
        'n in S'.

        Examples
        --------
        >>> S = PathGraph()
        >>> S.add_path([0,1,2,3])
        >>> 1 in S
        True
        """
        try:
            return n in self.node
        except TypeError:
            return False

    def __len__(self):
        """Return the number of nodes. Use the expression 'len(S)'.

        Returns
        -------
        nnodes : int
            The number of nodes in the graph.

        Examples
        --------
        >>> S = PathGraph()
        >>> S.add_path([0,1,2,3])
        >>> len(S)
        4

        """
        return len(self.node)

    def __getitem__(self, n):
        """Returns the path contains node n. Use the expression 'S[n]'.
        If n does not belong to a path, return [n]

        Parameters
        ----------
        n : node
           A node in the graph.

        Returns
        -------
        adj_list : list
           The adjacency list for nodes connected to n.

        Notes
        -----
        Assigning S[n] will corrupt the internal graph data structure.
        Use S[n] for reading data only.

        Examples
        --------
        >>> S = PathGraph()
        >>> S.add_path([0,1,2,3])
        >>> S[0]
        [0, 1, 2, 3]
        >>> S.add_node(4)
        >>> S[4]
        [4]
        """
        if n not in self.node:
            raise PathGraphNodeUnknown('Node {} does not exists'.format(n))
        else:
            if n not in self.path_id:
                return [n]
            else:
                return self.path[self.path_id[n]]

    def add_node(self, n, path_id=None, attr_dict=None, **attr):
        """Add a single node n and update node attributes.

        Parameters
        ----------
        n : node
            A node can be any hashable Python object except None.
        path_id : int the path id to which the node belongs
        attr_dict : dictionary, optional (default= no attributes)
            Dictionary of node attributes.  Key/value pairs will
            update existing data associated with the node.

        Examples
        --------
        >>> S = PathGraph()
        >>> S.add_node('C1')
        >>> S['C1']
        ['C1']

        Use keywords set/change node attributes:

        >>> S.add_node('C1', length=3000)
        >>> S.add_node('C2', length=400, color='red')

        >>> S.add_node('C3', path_id=0)
        Traceback (most recent call last):
        ...
        PathGraphException: Path id: 0 does not exists

        >>> S.add_node('C1')
        >>> S.path_id['C1']
        Traceback (most recent call last):
        ...
        KeyError: 'C1'

        >>> S.add_node('C2')
        >>> S.add_node('C3')
        >>> S.add_path(['C1', 'C2', 'C3'])
        >>> S.path_id['C1']
        0

        Notes
        -----
        A hashable object is one that can be used as a key in a Python
        dictionary. This includes strings, numbers, tuples of strings
        and numbers, etc.

        """

        # set up attribute dict
        if attr_dict is None:
            attr_dict = attr
        else:
            try:
                attr_dict.update(attr)
            except AttributeError:
                raise PathGraphException("The attr_dict argument must be a dictionary.")

        if n not in self.node:
            self.node[n] = attr_dict
            self.adj[n] = {}

        else:  # update attr even if node already exists
            self.node[n].update(attr_dict)

        if path_id is not None:
            # check that the referred path exists and contains node
            if path_id in self.path:
                if n not in self.path[path_id]:
                    raise PathGraphException("Node {} is not in path {}".format(n, path_id))
            else:
                raise PathGraphException("Path id: {} does not exists".format(path_id))

            self.path_id[n] = path_id

    def delete_node(self, n):
        """
        Remove node n.
        Removes the node n and all adjacent edges. If the node belongs to a path
        the path will be split at the node position
        Attempting to remove a non-existent node will raise an exception.

        Parameters
        ----------
        n : node
           A node in the graph

        Raises
        -------
        PathGraphError
           If n is not in the graph.

        Examples
        --------

        >>> S = PathGraph()
        >>> S.add_node('C1')
        >>> S['C1']
        ['C1']

        >>> S.delete_node('C1')
        >>> S['C1']
        Traceback (most recent call last):
        ...
        PathGraphNodeUnknown: Node C1 does not exists

        Check that the node is deleted from a path
        and that the path is split
        >>> S.add_path([0,1,2,3], name='path_1')
        >>> S.delete_node(2)
        >>> S.path
        {'path_1/1': [0, 1], 'path_1/2': [3]}
        """

        if n not in self.node:
            raise PathGraphException("The node {} is not in the graph.".format(n))

        self.delete_node_from_path(n)
        self.delete_path_containing_node(n)
        del self.node[n]

    def add_path(self, nodes, name=None, attr_dict=None, **attr):
        """Add a path consisting of the ordered nodes

        Parameters
        ----------
        nodes : iterable container
            A container of nodes.  A path will be constructed from
            the nodes (in order) and added to the graph.

        attr_dict : dictionary, optional (default= no attributes)
            Dictionary of edge attributes.  Key/value pairs will
            update existing data associated with the edge.

        attr : keyword arguments, optional
            Edge data (or labels or objects) can be assigned using
            keyword arguments.


        Examples
        --------
        >>> S = PathGraph()
        >>> S.add_path([0,1,2,3], name='path_1')
        >>> S[1]
        [0, 1, 2, 3]
        >>> S.path
        {'path_1': [0, 1, 2, 3]}

        >>> S.add_path(['c1', 'c1', 'c2'])
        Traceback (most recent call last):
        ...
        PathGraphException: Path contains repeated elements. Can't add path

        >>> S = PathGraph()
        >>> S.add_path([0,1,2,3], weight=1000)
        >>> S.adj[1]
        {0: {'weight': 1000}, 2: {'weight': 1000}}
        """
        if len(nodes) == 0:
            raise PathGraphException("The path is empty")

        if attr_dict is None:
            attr_dict = attr
        else:
            try:
                attr_dict.update(attr)
            except AttributeError:
                raise PathGraphException("The attr_dict argument must be a dictionary.")

        nlist = list(nodes)
        # check that the nodes do not belong to other path
        seen = set()
        for node in nlist:
            if node in self.path_id:
                raise PathGraphException("Node {} already belongs to another path. Can't add path".format(node))
            seen.add(node)

        if len(nodes) != len(seen):
            raise PathGraphException("Path contains repeated elements. Can't add path")

        if name is not None:
            if name not in self.path:
                path_id = name
            else:
                raise PathGraphException("Path name already exists {}".format(name))
        else:
            # get a new path_id
            path_id = len(self.path)
            if path_id in self.path:
                for index in range(len(self.path) + 1):
                    if index not in self.path:
                        path_id = index
                    break

        assert path_id not in self.path, "*Error*, path_id already in path"
        self.path[path_id] = nodes

        # add nodes
        for node in nlist:
            self.add_node(node, path_id=path_id)

        for i in range(len(nlist)-1):
            u = nlist[i]
            v = nlist[i+1]
            # add the edges
            datadict = self.adj[u].get(v, {})
            datadict.update(attr_dict)
            self.adj[u][v] = datadict
            self.adj[v][u] = datadict

    def get_path_name_of_node(self, node):
        """
        Returns the path name of the node
        Parameters
        ----------
        node

        Returns
        -------
        string path name of the node

        """

        if node in self.path_id:
            name = self.path_id[node]
        else:
            name = self.node[node]['name']
        return name

    def add_edge(self, u, v, name=None, return_direction=False, attr_dict=None, **attr):
        """
        Given a node u and a node v, this function appends and edge between u and v.
        Importantly, the function checks that this operation is possible. If u is
        not at the edge of a path, then a new node can not be added.

        Parameters
        ----------
        u : node id
        v : node id
        attr_dict : dictionary, optional (default= no attributes)
            Dictionary of edge attributes.  Key/value pairs will
            update existing data associated with the edge.

        attr : keyword arguments, optional
            Edge data (or labels or objects) can be assigned using
            keyword arguments.

        Returns
        -------
        direction_u, direction_v as for example ("+", "-")

        Examples
        --------

        >>> S = PathGraph()
        >>> S.add_path([0,1,2])
        >>> S.add_node(3)
        >>> S.add_edge(2, 3)
        >>> S[2]
        [0, 1, 2, 3]

        # check that second path is inverted
        >>> S.add_path([6, 5, 4])
        >>> S.add_edge(3,4)
        >>> S[4]
        [0, 1, 2, 3, 4, 5, 6]

        # check that first path is inverted
        >>> S = PathGraph()
        >>> S.add_path([2, 1,0], name='a')
        >>> S.add_path([3, 4, 5], name='b', weight=3)
        >>> S.path == {'a': [2, 1, 0], 'b': [3, 4, 5]}
        True
        >>> S.add_edge(2,3, weight=10)
        >>> S[4]
        [0, 1, 2, 3, 4, 5]

        >>> S.path == {'a, b': [0, 1, 2, 3, 4, 5]}
        True
        >>> S.adj[3]
        {2: {'weight': 10}, 4: {'weight': 3}}

        >>> S.add_edge(0, 5)
        Traceback (most recent call last):
        ...
        PathGraphEdgeNotPossible: Joining nodes 0, 5 forms a circle

        Test return direction
        >>> S = PathGraph()
        >>> S.add_path([0, 1, 2, 3])
        >>> S.add_path([6, 5, 4])
        >>> S.add_edge(3,4, return_direction=True)
        ('+', '-')

        >>> S.add_path([7, 8])
        >>> S.add_edge(7, 6, return_direction=True)
        ('-', '-')

        >>> S[7]
        [8, 7, 6, 5, 4, 3, 2, 1, 0]

        """
        if attr_dict is None:
            attr_dict = attr
        else:
            try:
                attr_dict.update(attr)
            except AttributeError:
                raise PathGraphException("The attr_dict argument must be a dictionary.")

        path = {}
        for node in [u, v]:
            # check if the node exists
            if node not in self.node:
                message = "Can't add edge {}-{}. Node {} does not exists ".format(u, v, node)
                raise PathGraphException(message)

            # check if the nodes are flanking a path
            path[node] = self[node]
            if node not in [path[node][0], path[node][-1]]:
                message = "Can't add edge {}-{}. Node {} does not flank its path {} ".format(u, v, node, path[node])
                raise PathGraphEdgeNotPossible(message)

        # check that nodes are not already in the same path
        try:
            if self.path_id[u] == self.path_id[v]:
                raise PathGraphEdgeNotPossible("Joining nodes {}, {} forms a circle".format(u, v))
        except KeyError:
            pass
        # the idea is to join nodes u,v such that the
        # final path is [...., u, v, ...]
        # for this, the paths containing u and v has to be
        # oriented properly

        # if u is the start of a path
        # invert the direction of the path
        direction_u = "+"
        if len(path[u]) and path[u][0] == u:
            path[u] = path[u][::-1]
            direction_u = "-"

        # if v is at the end of the path
        # invert the direction of the path
        direction_v = "+"
        if len(path[v]) and path[v][-1] == v:
            path[v] = path[v][::-1]
            direction_v = "-"

        if name is None:
            # get as name for the new path, a combination
            # of the merged paths names
            new_name = []
            for node in [u, v]:
                new_name.append(self.get_path_name_of_node(node))
            new_name = ", ".join(map(str, new_name))
        else:
            new_name = None

        # remove previous path and add new path
        for node in [u, v]:
            self.delete_path_containing_node(node, keep_adj=True)

        self.add_path(path[u] + path[v], name=new_name)
        datadict = self.adj[u].get(v, {})
        datadict.update(attr_dict)
        self.adj[u][v] = datadict
        self.adj[v][u] = datadict

        if return_direction:
            return direction_u, direction_v

    def delete_node_from_path(self, n):
        """
        removes the given node from a path. The path is split into two unless
        the node was the first or last.

        Parameters
        ----------
        n : node name

        Returns
        -------
        None

        Examples
        --------

        >>> S = PathGraph()
        >>> S.add_node("C1")

        C1 does not belong to a path so nothing should happen
        >>> S.delete_node_from_path('C1')


        >>> S.add_path([0,1,2,3])
        >>> S.delete_node_from_path(2)
        >>> S.path
        {0: [0, 1], 1: [3]}

        Test named paths
        >>> S = PathGraph()
        >>> S.add_path([0,1,2,3,4], name='path_1')
        >>> S.delete_node_from_path(2)
        >>> S.path
        {'path_1/1': [0, 1], 'path_1/2': [3, 4]}

        Test first and last node in path
        >>> S.delete_node_from_path(0)
        >>> S.path
        {'path_1/1': [1], 'path_1/2': [3, 4]}

        >>> S.delete_node_from_path(4)
        >>> S.path
        {'path_1/1': [1], 'path_1/2': [3]}
        """
        if n not in self.node:
            raise PathGraphException("The node {} is not in the graph.".format(n))

        if n not in self.path_id:
            return
        path_id = self.path_id[n]

        if len(self.path[path_id]) == 1:
            assert self.adj[n] == {}
            del self.path[n]
            return

        idx_n = self.path[path_id].index(n)
        new_path_left = self.path[path_id][:idx_n]
        new_path_right = self.path[path_id][idx_n+1:]

        # delete original path
        self.delete_path_containing_node(n)
        if not new_path_left:
            self.add_path(new_path_right, name=path_id)
        elif not new_path_right:
            self.add_path(new_path_left, name=path_id)
        else:
            new_name_left, new_name_right = PathGraph.new_split_path_names(path_id)
            self.add_path(new_path_left, name=new_name_left)
            self.add_path(new_path_right, name=new_name_right)
        del self.adj[n]

    def delete_edge(self, u, v):
        """

        Parameters
        ----------
        u
        v

        Returns
        -------

        >>> S = PathGraph()
        >>> S.add_path([0,1,2,3])
        >>> S[0]
        [0, 1, 2, 3]
        >>> S.delete_edge(1,2)
        >>> S.path
        {0: [0, 1], 1: [2, 3]}

        # renaming of named paths
        >>> S = PathGraph()
        >>> S.add_path([0,1,2,3], name='test')
        >>> S.delete_edge(1,2)
        >>> S.path == {'test/1': [0, 1], 'test/2': [2, 3]}
        True
        """
        def get_name_from_paths(new_path):
            new_name = []
            prev_name = None
            for node in new_path:
                name = self.node[node]['name']
                if prev_name is not None and prev_name != name:
                    new_name.append(name)
                prev_name = name

            if prev_name is not None:
                new_name.append(name)

            return ", ".join(new_name)

        # check that u and v are in the same path
        if self.path_id[u] != self.path_id[v]:
            message = "Can remove edge between {} and {} because they do not belong to the same path".format(u, v)
            raise PathGraphException(message)

        # check that u and v are directly connected
        if u not in self.adj[v]:
            message = "Can remove edge between {} and {} because they are not directly connected".format(u, v)
            raise PathGraphException(message)

        path_id = self.path_id[u]
        idx_u = self.path[path_id].index(u)
        idx_v = self.path[path_id].index(v)

        if idx_u > idx_v:
            idx_u, idx_v = idx_v, idx_u
        new_path_u = self.path[path_id][:idx_v]
        new_path_v = self.path[path_id][idx_v:]

        # this conditions happen when the nodes belong to two different paths
        # and the new path names is the split of the two paths
        if 'name' in self.node[u] and 'name' in self.node[v] and self.node[u]['name'] != self.node[v]['name']:
            new_name_u = get_name_from_paths(new_path_u)
            new_name_v = get_name_from_paths(new_path_v)
        else:
            new_name_u, new_name_v = PathGraph.new_split_path_names(path_id)

        # delete path
        self.delete_path_containing_node(u)

        # add the new two paths
        self.add_path(new_path_u, name=new_name_u)
        self.add_path(new_path_v, name=new_name_v)

    def delete_path_containing_node(self, n, keep_adj=False, delete_nodes=False):
        """

        Parameters
        ----------
        n : path containing node id is deleted
        keep_adj : if set to True, then the self.adj values are not removed. It is practical to
                   keep the adj when a path simply updated because of add edge or remove edge
        delete_nodes : bool if True, not only the nodes but the path are deleted
        Returns
        -------


        Examples
        --------

        >>> S = PathGraph()
        >>> S.add_path([0,1,2,3])
        >>> S[1]
        [0, 1, 2, 3]

        >>> S.delete_path_containing_node(2)
        >>> S.path
        {}
        >>> S.path_id
        {}

        >>> S.adj[1]
        {}

        Test delete_nodes=True
        >>> S = PathGraph()
        >>> S.add_path([0,1,2,3])
        >>> S.delete_path_containing_node(2, delete_nodes=True)
        >>> S.node
        {}

        >>> S = PathGraph()
        >>> S.add_node("test")
        >>> S.delete_path_containing_node("test", delete_nodes=True)
        >>> S.node
        {}
        """

        if n not in self.path_id and delete_nodes:
            del self.node[n]
            del self.adj[n]

        if n in self.path_id:
            path_id = self.path_id[n]
            for _n in self.path[self.path_id[n]]:
                del self.path_id[_n]
                if not keep_adj:
                    self.adj[_n] = {}
                if delete_nodes:
                    del self.node[_n]
                    del self.adj[_n]
            del self.path[path_id]

    def merge_paths(self, paths):
        """
        Given a list of paths, those are merged into a larger path
        Parameters
        ----------
        paths: list of paths

        Returns
        -------

        """
        merged_path = sum(paths, [])
        if len(merged_path) != len(set(merged_path)):
            raise PathGraphException("Path contains repeated elements. Can't add path")

        # delete paths to merge
        for path in paths:
            self.delete_path_containing_node(path[0])

        self.add_path(merged_path)

    def get_all_paths(self):
        """Returns all paths in the graph.
        This is similar to get connected components in networkx
        but in this case, the order of the returned  paths
        represents scaffolds of contigs

        >>> S = PathGraph()
        >>> S.add_path([0, 1, 2])
        >>> S.add_path([3, 4])
        >>> S.add_node(5)
        >>> list(S.get_all_paths())
        [[0, 1, 2], [3, 4], [5]]
        """
        seen = set()
        for v in self:
            # v in self returns all nodes in the pathgraph
            if v not in seen:
                # self [v] returns a path containing v. If the v does not belong to a path
                # a singleton path [v] is returned
                yield self[v]
            seen.update(self[v])

    @staticmethod
    def new_split_path_names(path_id):
        if isinstance(path_id, int):
            new_name_u = None
            new_name_v = None
        else:
            # for paths with names, assign a new name adding
            # `/n`
            # by adding
            # check if the path was divided before
            res = re.search("(.*?)/(\d+)$", str(path_id))
            if res is not None:
                path_id = res.group(1)
                split_number = int(res.group(2))
            else:
                split_number = 1
                path_id = str(path_id)
            new_name_u = str(path_id) + "/{}".format(split_number)
            new_name_v = str(path_id) + "/{}".format(split_number + 1)

        return new_name_u, new_name_v


class PathGraphException(Exception):
    """Base class for exceptions in PathGraph"""


class PathGraphNodeUnknown(PathGraphException):
    """Exception when trying to access an unknown node"""


class PathGraphEdgeNotPossible(PathGraphException):
    """Exception when trying to add an edge where one of the nodes is inside a path or when the edge forms a loop"""
