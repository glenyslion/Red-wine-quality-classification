filename = r"C:\Users\Glenys Charity Lion\Desktop\train.csv"
fields = []

with open(filename,'r') as csvreader:
    fix_acid, volatile_acid, citric_acid, residual, chlorides, free_sulfur, total_sulfur, density, pH, sulphates, alcohol, quality = [], [], [], [], [], [], [], [], [], [], [], []
    fields = next(csvreader)
    # Convert csv format to a list based on its column
    for row in csvreader:
        fa = row.split(',')
        fix_acid.append(float(fa[0]))
        volatile_acid.append(float(fa[1]))
        citric_acid.append(float(fa[2]))
        residual.append(float(fa[3]))
        chlorides.append(float(fa[4]))
        free_sulfur.append(float(fa[5]))
        total_sulfur.append(float(fa[6]))
        density.append(float(fa[7]))
        pH.append(float(fa[8]))
        sulphates.append(float(fa[9]))
        alcohol.append(float(fa[10]))
        quality.append(float(fa[11]))

    # 1 means quality above 6, 0 means quality less than or equal to 6
    for x in range(len(quality)):
        if(quality[x] > 6):
            quality[x] = 1
        else:
            quality[x] = 0

    class Node:
        # Initialize the node class
        def __init__(self,gini,num_sample,num_sample_per_class,predicted_class):
            self.gini = gini
            self.num_sample = num_sample
            self.num_sample_per_class = num_sample_per_class
            self.predicted_class = predicted_class
            self.feature_index = 0
            self.threshold = 0
            self.left = None
            self.right = None

    class DecisionTreeClassifier:
        # Initialize the decision tree classifier
        def __init__(self, max_depth=None):
            self.max_depth = max_depth
        
        # Find the gini of a node
        def gini(self, y):
            length = len(y)
            count_1 = 0
            count_0 = 0
            for x in y:
                if x == 1:
                    count_1 += 1
                else:
                    count_0 += 1
            return 1.0 - (count_1/length)**2 - (count_0/length)**2

        # Find the best split of a node (gini after split)
        def best_split(self, x, y):
            best_gini = self.gini(y)
            best_threshold = 1
            best_num = 0
            # The number of features = 11 (Find gini after split of each feature and output best predictor and threshold)
            for num in range(11):
                listt = []
                for row in x:
                    listt.append(row[num])
                for threshold in listt:
                    temp = []
                    for a in listt:
                        if a > threshold:
                            temp.append(1)
                        else:
                            temp.append(0)
                    gini = self.count_gini(temp, y)
                    # If gini better than previous best_gini, update the best_gini, best_num, and best_threshold
                    if(gini < best_gini):
                        best_gini = gini
                        best_threshold = threshold
                        best_num = num
            return best_num, best_threshold

        # Compute the gini impurity of non-empty node
        def count_gini(self, temp, y):
            # yes_1 = number of records where y = 1 and x > threshold
            # yes_0 = number of records where y = 1 and x <= threshold
            yes_1 = 0
            yes_0 = 0
            # count_1 = number of records where x > threshold
            # count_0 = number of records where x <= threshold
            count_1 = 0
            count_0 = 0
            # Count gini
            for x in range(len(temp)):
                if(temp[x] == 1):
                    count_1 += 1
                    if(y[x] == 1):
                        yes_1 += 1
                else:
                    count_0 += 1
                    if(y[x] == 1):
                        yes_0 += 1
            if (count_1 != 0):
                gini_1 = 1-(yes_1/count_1)**2 - ((count_1-yes_1)/count_1)**2
            else:
                gini_1 = 0
            if(count_0!=0):
                gini_0 = 1-(yes_0/count_0)**2 - ((count_0-yes_0)/count_0)**2
            else:
                gini_0 = 0
            gini = (count_1/(count_1+count_0))*gini_1 + (count_0/(count_1+count_0)) * gini_0
            return gini

        # Fitting the predictors for y (quality) in the train data
        def fit(self,x,y):
            self.tree = self.grow_tree(x, y)

        # Build the tree recursively
        def grow_tree(self,x,y,depth=0):
            count_0 = 0
            count_1 = 0
            num_sample_per_class = []
            for a in y:
                if(a==0):
                    count_0 += 1
                else:
                    count_1 += 1
            num_sample_per_class.append(count_0)
            num_sample_per_class.append(count_1)
            # Predict based on the probability (pick the class with higher probability)
            if(count_0 > count_1):
                predicted_class = 0
            else:
                predicted_class = 1
            node = Node(gini=self.gini(y),num_sample=len(y), num_sample_per_class=num_sample_per_class,predicted_class=predicted_class)
            # If count_0 = 0 or count_1 = 0, it can't be improved (gini = 0)
            if count_0 == 0 or count_1 == 0:
                return node
            # Continue recursively if depth is less than the maximum depth allowed
            if depth < self.max_depth:
                idx, thr = self.best_split(x,y)
                if idx is not None:
                    x = sorted(x, key = lambda l:l[idx])
                    indices_left = 0
                    for i in range(len(x)):
                        if x[i][idx] > thr:
                            indices_left = i
                            break
                    if indices_left == 0 or indices_left == len(x)-1:
                        return node
                    x_left,y_left = x[:indices_left],y[:indices_left]
                    x_right,y_right = x[indices_left:len(x)],y[indices_left:len(x)]
                    node.feature_index = idx
                    node.threshold = thr
                    node.left = self.grow_tree(x_left,y_left,depth+1)
                    node.right = self.grow_tree(x_right,y_right,depth+1)
            return node

        # Predict class of x, print the actual and prediction result, and count the correct predictions
        def predict(self,x):
            correct = 0
            for b in range (len(x)):
                print("ACTUAL = ",quality2[b],"PREDICTION = ",self.PREDICT(x[b]))
                if(quality2[b]==self.PREDICT(x[b])):
                    correct += 1
            return correct

        # Predict class of a single sample
        def PREDICT(self, inputs):
            node = self.tree
            while node.left:
                if inputs[node.feature_index] < node.threshold:
                    node = node.left
                else:
                    node = node.right
            return node.predicted_class

    count = DecisionTreeClassifier(11)
    matrix = []
    # Use for loop to make a list based on its row and the result will be list inside a list
    for x in range(len(fix_acid)):
        lst = []
        lst.append(fix_acid[x])
        lst.append(volatile_acid[x])
        lst.append(citric_acid[x])
        lst.append(residual[x])
        lst.append(chlorides[x])
        lst.append(free_sulfur[x])
        lst.append(total_sulfur[x])
        lst.append(density[x])
        lst.append(pH[x])
        lst.append(sulphates[x])
        lst.append(alcohol[x])
        matrix.append(lst)

    test_data = r"C:\Users\Glenys Charity Lion\Desktop\test.csv"
    fields_test = []
    with open(test_data, 'r') as csvreader:
        fix_acid2, volatile_acid2, citric_acid2, residual2, chlorides2, free_sulfur2, total_sulfur2, density2, pH2, sulphates2, alcohol2, quality2 = [], [], [], [], [], [], [], [], [], [], [], []
        fields_test = next(csvreader)

        # Convert csv format to a list based on its column
        for row in csvreader:
            fu = row.split(',')
            fix_acid2.append(float(fu[0]))
            volatile_acid2.append(float(fu[1]))
            citric_acid2.append(float(fu[2]))
            residual2.append(float(fu[3]))
            chlorides2.append(float(fu[4]))
            free_sulfur2.append(float(fu[5]))
            total_sulfur2.append(float(fu[6]))
            density2.append(float(fu[7]))
            pH2.append(float(fu[8]))
            sulphates2.append(float(fu[9]))
            alcohol2.append(float(fu[10]))
            quality2.append(float(fu[11]))
        
        for b in range(len(quality2)):
            if(quality2[b] > 6):
                quality2[b] = 1
            else:
                quality2[b] = 0

        matrix2 = []
        # Use for loop to make a list based on its row and the result will be list inside a list
        for x in range(len(fix_acid2)):
            lst2 = []
            lst2.append(fix_acid2[x])
            lst2.append(volatile_acid2[x])
            lst2.append(citric_acid2[x])
            lst2.append(residual2[x])
            lst2.append(chlorides2[x])
            lst2.append(free_sulfur2[x])
            lst2.append(total_sulfur2[x])
            lst2.append(density2[x])
            lst2.append(pH2[x])
            lst2.append(sulphates2[x])
            lst2.append(alcohol2[x])
            matrix2.append(lst2)
    # Build the tree
    count.fit(matrix, quality)
    # Return value of the correct prediction
    correct_prediction = count.predict(matrix2)
    # Length of quality
    total_len = len(quality2)
    # Count the accuracy based on the equation given in the assignment
    accuracy = correct_prediction/total_len
    # Print the accuracy
    print("ACCURACY = ",accuracy)
