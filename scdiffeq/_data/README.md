### API overview of `sdq.data`

To load the fully pre-processed dataset:
* here we set `force=True` for illustrative purposes. 

```python
adata = sdq.data.Weinreb2020_preprocessed(force=True)
```
<img width="1016" alt="Screen Shot 2022-02-26 at 11 18 26 PM" src="https://user-images.githubusercontent.com/47393421/155868037-37645ba3-ac2b-4b7d-b55a-f27fcd8f9791.png">


To plot the dataset as shown in the paper:
    
```python
sdq.data.plotWeinreb2020(adata)
```
![image](https://user-images.githubusercontent.com/47393421/155867997-672028ab-f1f3-4c18-b7bb-b184101f02bc.png)