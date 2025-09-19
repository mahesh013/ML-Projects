import sys
import os
import pandas as pd

# keep your custom exception and loader
from src.exception import CustomException
from src.utils import load_object

# try import OneHotEncoder and containers from sklearn
try:
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    import sklearn
except Exception:
    # If sklearn isn't available (shouldn't happen at runtime), we'll still proceed and surface errors later.
    OneHotEncoder = None
    ColumnTransformer = None
    Pipeline = None
    sklearn = None


class PredictPipeline:
    def __init__(self):
        pass

    def _patch_ohe_recursive(self, obj):
        """
        Recursively search for OneHotEncoder instances within Pipelines / ColumnTransformers
        and set missing internal attributes that some sklearn versions expect.
        This is a temporary workaround — prefer matching sklearn versions or re-saving the preprocessor.
        """
        if OneHotEncoder is None:
            return

        # If obj itself is a OneHotEncoder, patch it
        if isinstance(obj, OneHotEncoder):
            if not hasattr(obj, "_drop_idx_after_grouping"):
                setattr(obj, "_drop_idx_after_grouping", None)
            if not hasattr(obj, "drop_idx_"):
                setattr(obj, "drop_idx_", None)
            return

        # ColumnTransformer: traverse its transformers_
        if ColumnTransformer is not None and isinstance(obj, ColumnTransformer):
            try:
                for name, trans, cols in obj.transformers_:
                    # some transformers can be 'drop' or 'passthrough'
                    if trans is None or isinstance(trans, str):
                        continue
                    # if transformer is tuple/list (like a Pipeline), pass it to recursion
                    self._patch_ohe_recursive(trans)
            except Exception:
                # defensive: some older sklearn versions or unusual objects may not have transformers_
                pass
            return

        # Pipeline: traverse its steps
        if Pipeline is not None and isinstance(obj, Pipeline):
            try:
                for step_name, step in obj.steps:
                    self._patch_ohe_recursive(step)
            except Exception:
                pass
            return

        # Generic container possibility: if object has attribute 'transformers_' or 'steps', try them
        try:
            if hasattr(obj, "transformers_"):
                for t in getattr(obj, "transformers_"):
                    # t may be (name, transformer, columns)
                    if isinstance(t, (list, tuple)) and len(t) >= 2:
                        self._patch_ohe_recursive(t[1])
            if hasattr(obj, "steps"):
                for s in getattr(obj, "steps"):
                    if isinstance(s, (list, tuple)) and len(s) >= 2:
                        self._patch_ohe_recursive(s[1])
        except Exception:
            # swallow errors from unknown nested structures
            pass

    def predict(self, features):
        try:
            model_path = os.path.join("artifacts", "model.pkl")
            preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")

            print("Before Loading")
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            print("After Loading")

            # print sklearn version (useful for debugging version mismatch)
            try:
                print("sklearn version:", sklearn.__version__)
            except Exception:
                pass

            # First attempt to transform normally
            try:
                data_scaled = preprocessor.transform(features)
            except Exception as transform_exc:
                # If the transform failed due to the OneHotEncoder internal attribute,
                # attempt to patch any OneHotEncoder instances and retry once.
                err_msg = str(transform_exc)
                if "_drop_idx_after_grouping" in err_msg or "drop_idx_" in err_msg or "OneHotEncoder" in err_msg:
                    try:
                        print("Encountered OneHotEncoder internal attribute error — attempting to patch encoder objects and retry transform.")
                        self._patch_ohe_recursive(preprocessor)
                        data_scaled = preprocessor.transform(features)  # retry once
                    except Exception as retry_exc:
                        # If retry fails, raise the original exception wrapped in CustomException
                        raise CustomException(retry_exc, sys)
                else:
                    # Not the specific OHE attribute issue — rethrow wrapped
                    raise CustomException(transform_exc, sys)

            preds = model.predict(data_scaled)
            return preds

        except Exception as e:
            # Wrap and raise so your existing error handling works
            raise CustomException(e, sys)


class CustomData:
    def __init__(
        self,
        gender: str,
        race_ethnicity: str,
        parental_level_of_education,
        lunch: str,
        test_preparation_course: str,
        reading_score: int,
        writing_score: int,
    ):

        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "gender": [self.gender],
                "race_ethnicity": [self.race_ethnicity],
                "parental_level_of_education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test_preparation_course": [self.test_preparation_course],
                "reading_score": [self.reading_score],
                "writing_score": [self.writing_score],
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)
