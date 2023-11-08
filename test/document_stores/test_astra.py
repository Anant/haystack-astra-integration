import os
from inspect import getmembers, isclass, isfunction
from typing import Any, Dict, List, Union
from unittest.mock import MagicMock

import numpy as np
import pytest

from haystack.document_stores.astra import (
    AstraDocumentStore
)
from haystack.errors import FilterError, AstraDocumentStoreError
from haystack.schema import Document
from haystack.testing import DocumentStoreBaseTestAbstract

META_FIELDS = ["meta_field", "name", "date", "numeric_field", "odd_document", "doc_type"]


class TestAstraDocumentStore(DocumentStoreBaseTestAbstract):
    # Fixtures
    pass