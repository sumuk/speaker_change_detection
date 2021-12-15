from pyannote.core import Segment, Timeline, Annotation
from pyannote.metrics.segmentation import SegmentationCoverage,SegmentationPurity,SegmentationPurityCoverageFMeasure
from pyannote.metrics.segmentation import SegmentationRecall,SegmentationPrecision

class Metrics():
    '''
    class to calculate the metrics like recall and precision of the segmentation
    '''
    def __init__(self,tolerance=0.5):
        '''
        tolernace: pyannonte uses this when the decision boundary is within this time then the output is considered right
        '''
        self.coverage = SegmentationCoverage()
        self.purity = SegmentationPurity()
        self.f1 = SegmentationPurityCoverageFMeasure()
        self.re = SegmentationRecall(tolerance=tolerance)
        self.pr = SegmentationPrecision(tolerance=tolerance)
    
    def get_precision(self,ref,hyp):
        '''
        calculate the precision of the output
        '''
        assert isinstance(ref,Annotation),'reference needs to be of type annotation'
        assert isinstance(hyp,Annotation),'hypothesis needs to be of type annotation'
        return self.pr(ref,hyp)
    
    def get_recall(self,ref,hyp):
        '''
        calculate the recall of the output
        '''
        assert isinstance(ref,Annotation),'reference needs to be of type annotation'
        assert isinstance(hyp,Annotation),'hypothesis needs to be of type annotation'
        return self.re(ref,hyp)
    
    def get_coverage(self,ref,hyp):
        '''
        calculate the coverage of the output
        '''
        assert isinstance(ref,Annotation),'reference needs to be of type annotation'
        assert isinstance(hyp,Annotation),'hypothesis needs to be of type annotation'
        return self.coverage(ref,hyp)
    
    def get_purity(self,ref,hyp):
        '''
        calculate the purity of the output
        '''
        assert isinstance(ref,Annotation),'reference needs to be of type annotation'
        assert isinstance(hyp,Annotation),'hypothesis needs to be of type annotation'
        return self.purity(ref,hyp)
    
    def get_overall(self,ref,hyp):
        '''
        calculate the overall f1 measure wrt of purity and coverage of the output
        '''
        assert isinstance(ref,Annotation),'reference needs to be of type annotation'
        assert isinstance(hyp,Annotation),'hypothesis needs to be of type annotation'
        return self.f1(ref,hyp)