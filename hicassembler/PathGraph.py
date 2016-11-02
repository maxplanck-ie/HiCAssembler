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

    def __iter__(self):
        """Iterate over the nodes. Use the expression 'for n in S'.

        Returns
        -------
        niter : iterator
            An iterator over all nodes in the graph.

        Examples
        --------
        >>> S = PathGraph()
        >>> S.add_path([0,1,2,3])
        """
        return iter(self.node)

    def __contains__(self,n):
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
        """Return a list of that contains node n.  Use the expression 'S[n]'.

        Parameters
        ----------
        n : node
           A node in the graph.

        Returns
        -------
        adj_dict : dictionary
           The adjacency dictionary for nodes connected to n.

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
        PathGraphNodeUnknown: 'C1'

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
            attr_dict=attr
        else:
            try:
                attr_dict.update(attr)
            except AttributeError:
                raise PathGraphException("The attr_dict argument must be a dictionary.")

        if n not in self.node:
            self.node[n] = attr_dict

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

    def add_path(self, nodes):
        """Add a path consisting of the ordered nodes

        Parameters
        ----------
        nodes : iterable container
            A container of nodes.  A path will be constructed from
            the nodes (in order) and added to the graph.


        Examples
        --------
        >>> S = PathGraph()
        >>> S.add_path([0,1,2,3])
        >>> S[1]
        [0, 1, 2, 3]

        >>> S.add_path(['c1', 'c1', 'c2'])
        Traceback (most recent call last):
        ...
        PathGraphException: Path contains repeated elements. Can't add path


        """
        nlist = list(nodes)
        # check that the nodes do not belong to other path
        seen = set()
        for node in nlist:
            if node in self.path_id:
                raise PathGraphException("Node {} already belongs to another path. Can't add path".format(node))
            seen.add(node)

        if len(nodes) != len(seen):
            raise PathGraphException("Path contains repeated elements. Can't add path")

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

    def add_edge(self, u, v):
        """
        Given a node u and a node v, this function appends and edge between u and v.
        Importantly, the function checks that this operation is possible. If u is
        not at the edge of a path, then a new node can not be added.

        Parameters
        ----------
        u : node id
        v : node id
        attr

        Returns
        -------
        None

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
        >>> S.add_path([2, 1,0])
        >>> S.add_path([3, 4, 5])
        >>> S.add_edge(2,3)
        >>> S[4]
        [0, 1, 2, 3, 4, 5]

        >>> S.add_edge(0, 5)
        Traceback (most recent call last):
        ...
        PathGraphException: Joining nodes 0, 5 forms a circle
        """
        path = {}
        for node in [u, v]:
            # check if the node exists
            if node not in self.node:
                message = "Can't add edge {}-{}. Node {} does not exists ".format(u, v, node)
                raise PathGraphException(message)

            # check if the nodes are flanking a path
            path[node] = self[node]
            if node not in [path[node][0], path[node][-1]]:
                message = "Can't add edge {}-{}. Node {} does not exists ".format(u, v, node)
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
        if len(path[u]) and path[u][0] == u:
            path[u] = path[u][::-1]

        # if v is at the end of the path
        # invert the direction of the path
        if len(path[v]) and path[v][-1] == v:
            path[v] = path[v][::-1]

        # remove previous path and add new path
        for node in [u, v]:
            self.delete_path_containing_node(node)

        self.add_path(path[u] + path[v])

    def delete_path_containing_node(self, n):
        """

        Parameters
        ----------
        n : path containing node id is deleted

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
        """
        if n in self.path_id:
            path_id = self.path_id[n]
            for _n in self.path[self.path_id[n]]:
                del self.path_id[_n]
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


class PathGraphException(Exception):
    """Base class for exceptions in PathGraph"""


class PathGraphNodeUnknown(PathGraphException):
    """Exception when trying to access an unknown node"""

class PathGraphEdgeNotPossible(PathGraphException):
    """Exception when trying to add an edge where one of the nodes is inside a path or when the edge forms a loop"""
