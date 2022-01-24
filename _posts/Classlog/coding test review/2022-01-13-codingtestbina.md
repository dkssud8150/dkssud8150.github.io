---
title:    "Coding Test[Python] - 이진트리 구현하기"
author:
  name: JaeHo-YooN
  link: https://github.com/dkssud8150
date: 2022-01-13 12:00:00 +0800
categories: [Classlog,coding test review]
tags: [coding test, binary-tree]
toc: True
comments: True
math: true
mermaid: true
#image:
#  src: /commons/devices-mockup.png
#  width: 800
#  height: 500
---

<br>

# 이진트리란?

이진트리는 각 노드의 자식 수가 2 이하인 트리이다. 이진트리는 4가지 순회방식이 존재하는데, 방식은 각각 다르지만, 항상 트리의 루트부터 시작한다는 점이 동일하다. 또한, 모든 순회방식이 트리의 모든 노드를 반드시 1번씩 방문한 후 종료된다. 순회방식은 다음과 같다.

* 전위순회(Preorder)
* 중위순회(Inorder)
* 후위순회(Postoder)
* 레벨순회(Level-order)

## 전위순회

전위순회의 순회 방식 NLR이고 이는 다음과 같다.
1. 현재 노드 n를 먼저 방문 (N = 현재 노드)
2. 현재 노드의 왼쪽 서브트리를 순회 (L = 왼쪽)
3. 현재 노드의 오른쪽 서브트리를 순회 (R = 오른쪽)

```python
def preoder(self, n):
    if n != None:
        print(n.item,'',end='')             # 노드 방문
        if n.left: self.preorder(n.left)    # 왼쪽 서브트리 순회
        if n.right: self.preorder(n.right)  # 오른쪽 서브트리 순회
```

<br>

## 중위순회

중위순회의 방식은 LNR로
1. 현재 노드의 왼쪽 서브트리를 순회 (L = 왼쪽)
2. 현재 노드 n를 먼저 방문 (N = 현재 노드)
3. 현재 노드의 오른쪽 서브트리를 순회 (R = 오른쪽)

```python
def inorder(self, n):
    if n != None:
        if n.left: self.inorder(n.left)
        print(n.item, '', end='')
        if n.right: self.inorder(n.right)
```

<br>

## 후위순회

후위순회의 방식은 LRN로
1. 현재 노드의 왼쪽 서브트리를 순회 (L = 왼쪽)
2. 현재 노드의 오른쪽 서브트리를 순회 (R = 오른쪽)
3. 현재 노드 n를 먼저 방문 (N = 현재 노드)

```python
def postorder(self, n):
    if n != None:
        if n.left: self.postorder(n.left)
        if n.right: self.postorder(n.right)
        print(n.item, '', end='')
```

<br>

## 레벨순회

레벨순회의 방식은 루트가 있는 곳부터 각 레벨마다 좌에서 우로 노드를 방문한다.

```python
def levelorder(self, root):
    q = []
    q.append(root)
    while q:
        t = q.pop(0)
        print(n.item,'',end='')                 # q에서 첫 항목을 삭제하고 삭제한 노드 방문
        if n.left != None: q.append(t.left)     # 왼쪽 자식 큐에 삽입
        if n.right != None: q.append(t.right)   # 오른쪽 자식 큐에 삽입
```

<br>

이진트리는 높이 h에 대해 최대 2^(h-1)만큼 노드를 가질 수 있다. 높이 h는 루트 노드를 기준으로 두 자식노드의 높이 중 큰 높이를 구하면 된다.

```python
def height(self, root):
    if root == None: return 0
    return max(self.height(root.left),self.height(root.right)) + 1
```

이처럼 방문을 한다는 것은 `print()`함수로 표현한다. 그래서 전위순회를 기준으로 예를 들어보면, root노드를 먼저 방문한 후, 왼쪽부터 순회를 돈다. 그러면 1번 내려가고, 거기에 있는 노드를 읽은 후 다시 왼쪽, 또 내려가서 왼쪽을 읽다가 자식이 없으면 1단계 위로 올라가서 오른쪽 노드를 읽고, 자식이 없으면 다시 위로 올라간다.

<br>

최종 코드는 다음과 같다.

```python
class Node:
    def __init__(self,item):
        self.item = item
        self.left = None
        self.right = None
    
class BinaryTree():
    def __init__(self):
        self.root = None
    
    #전위순회
    def preoder(self, n):
    if n != None:
        print(n.item,'',end='')       
        if n.left: self.preorder(n.left)   
        if n.right: self.preorder(n.right) 

    #중위순회
    def inorder(self, n):
    if n != None:
        if n.left: self.inorder(n.left)
        print(n.item, '', end='')
        if n.right: self.inorder(n.right)

    #후위순회
    def postorder(self, n):
    if n != None:
        if n.left: self.postorder(n.left)
        if n.right: self.postorder(n.right)
        print(n.item, '', end='')

    #레벨순회
    def levelorder(self, root):
    q = []
    q.append(root)
    while q:
        t = q.pop(0)
        print(n.item,'',end='')  
        if n.left != None: q.append(t.left)  
        if n.right != None: q.append(t.right) 

    #트리 높이
    def height(self, root):
        if root == None: return 0
        return max(self.height(root.left),self.height(root.right)) + 1  

tree = BinaryTree()
n1=Node(10);n2=Node(20);n3=Node(30);n4=Node(40);n5=Node(50);n6=Node(60);n7=Node(70);n8=Node(80);

#트리
tree.root = n1
n1.left=n2;n1.right=n3
n2.left=n4;n2.right=n5
n3.left=n6;n3.right=n7
n4.left=n8

print('트리 높이: {}\n전위순회: {}\n중위순회: {}\n후위순회: {}\n레벨순회: {}'.format(
    tree.height(tree.root),         # 트리 높이: 4
    tree.preorder(tree.root),       # 전위 순회: 10 20 40 80 50 30 60 70
    tree.inorder(tree.root),        # 중위 순회: 20 40 80 50 10 30 60 70
    tree.postorder(tree.order),     # 후위 순회: 20 40 80 50 30 60 70 10
    tree.levelorder(tree.order)))   # 레벨 순회: 10 20 30 40 50 60 70 80
```

*[참고 블로그1](https://it-garden.tistory.com/406)
*[참고 블로그2](https://brunch.co.kr/@qqplot/131)