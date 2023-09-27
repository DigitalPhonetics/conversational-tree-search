from typing import Dict, List, Optional, Union
import re
import json
import datetime
from pathlib import Path


resource_dir = Path("resources/en/reimburse")

with open(resource_dir / 'numbers.json', 'r') as f:
    NUMBERS = json.load(f)

with open(resource_dir / 'months.json', 'r') as f:
    MONTHS = json.load(f)

WEEKDAYS = {'monday': 1, 'tuesday': 2, 'wednesday': 3, 'thursday': 4, 'friday': 5, 'saturday': 6, 'sunday': 7}


class Number:

    def __init__(self, number_string: str, position_in_utterance: tuple, match_dict: Optional[dict] = None,
                 ordinal: bool = False):
        # Generic number class for all kind of numbers, number words, and time expressions recognized in an utterance
        self.original_string = number_string
        self.start_position = position_in_utterance[0]
        self.end_position = position_in_utterance[1]
        self.ordinal = ordinal

        self.unit = None  # possible: None, 'day', 'time', 'minutes', 'hours', 'days', 'weeks'
        self.time_type = None  # possible: None, 'point', 'span'; necessary for later conversion

        if match_dict is not None:  # in this case the slots for recognizing a number word (like 'zehn') are given
            self.number_value = self._convert_number_word(match_dict)
        elif self.original_string.isalpha():  # time words like 'gestern'
            self.number_value = self.original_string
            self.unit = 'day'
            self.time_type = 'point'
        else:  # everything else, like '10', '10.02.21', '10:35'
            self.number_value = self._convert_digit_number(self.original_string)
        self.value = self.number_value

    def add_unit(self, unit: str):
        # add the unit of the number and the time type (time point or time span) accordingly
        if unit in {'time', 'day'}:
            self.time_type = 'point'
            self.unit = unit
        elif unit in {'minutes', 'hours', 'days', 'weeks', 'months'}:
            self.time_type = 'span'
            self.unit = unit

    def add_month(self, value: str):
        # add the month to the number if we found it later than the number itself (e.g. when it is given as word,
        # like 'Dezember')
        month_value = MONTHS[value]
        self.number_value = (self.number_value, month_value)
        self.value = (self.value, month_value)
        self.time_type = 'point'
        self.unit = 'day'

    def __str__(self):
        return str(self.value)

    def __repr__(self):
        return str(self.value)

    def _convert_digit_number(self, digit_number: str):
        # convert a string number given in digits into an integer representation
        if '.' in digit_number:  # may be the case for dates and time expressions (e.g. 10.30 Uhr)
            splitted_number = digit_number.split('.')
            try:
                splitted_ints = [int(x) for x in splitted_number if x]
            except ValueError:
                return digit_number

            if len(splitted_ints) == 2:  # either a date with day and month (10.06.) or a time with hour and minute (10.30)
                if splitted_ints[0] < 32 and splitted_ints[1] < 60:  # this must be valid for being either date or time
                    self.time_type = 'point'
                    if splitted_ints[0] < 24 and splitted_ints[1] > 12:  # this cannot be a date, it must be a time
                        self.unit = 'time'
                        return tuple(splitted_ints)
                    else:  # if we are unsure whether it is time or date, we will regard it as date by default
                        self.unit = 'day'
                        return tuple(splitted_ints)

            elif len(splitted_ints) == 3:  # a date with day, month and year (like 10.06.2020)
                self.time_type = 'point'
                self.unit = 'day'
                return tuple(splitted_ints)

        elif ':' in digit_number:  # a time (like 10:30)
            splitted_number = digit_number.split(':')
            try:
                splitted_ints = [int(x) for x in splitted_number if x]
            except ValueError:
                return digit_number

            if len(splitted_ints) == 2:  # in this case it is clearly hour:minute, otherwise the string is not converted
                self.time_type = 'point'
                self.unit = 'time'
                return tuple(splitted_ints)

        elif digit_number.isnumeric():  # a simple number like '5'
            return int(digit_number)
        return digit_number

    def _convert_number_word(self, number_word_dict: Dict[str, str]):
        # convert a number given as word (e.g. 'dreißig') by using a dict indicating the matched slots
        number = 0
        if number_word_dict['thousand']:
            if number_word_dict['ones_thousand']:  # e.g. 'dreitausend'
                ones_number = NUMBERS[f'ones{"_ordinal" if self.ordinal else ""}'][number_word_dict[
                    'ones_thousand']]
                number += ones_number * 1000
            else:
                number += 1000
        if number_word_dict['hundred']:
            if number_word_dict['ones_hundred']:  # e.g. 'dreihundert'
                ones_number = NUMBERS[f'ones{"_ordinal" if self.ordinal else ""}'][number_word_dict[
                    'ones_hundred']]
                number += ones_number * 100
            else:
                number += 100
        if number_word_dict['tens']:  # e.g. 'zwanzig'
            number += NUMBERS[f'tens{"_ordinal" if self.ordinal else ""}'][number_word_dict['tens']]
        if number_word_dict['ones_with_tens']:  # e.g. in 'zweiundzwanzig'
            number += NUMBERS[f'ones{"_ordinal" if self.ordinal else ""}'][number_word_dict['ones_with_tens']]
        if number_word_dict['ones']:  # e.g. 'zwei'
            number += NUMBERS[f'ones{"_ordinal" if self.ordinal else ""}'][number_word_dict['ones']]
        if number_word_dict['single']:  # e.g. 'kein'
            number += NUMBERS[f'single'][number_word_dict['single']]
        return number


class TimePoint:

    def __init__(self, number_item: Optional[Number] = None,
                 time_object: Union[datetime.date, datetime.time, None] = None):
        # class for time points like dates and clock times
        assert number_item is not None or time_object is not None, \
            'Either a Number object must be given as number_item or a Datetime object (Time, Date) must be given as ' \
            'time_object!'
        if number_item:  # either read values from a Number item
            self.original_string = number_item.original_string
            self.start_position = number_item.start_position
            self.end_position = number_item.end_position
            self.ordinal = number_item.ordinal
            self.number_value = number_item.number_value
            self.unit = number_item.unit
            self.value = self._create_timestamp()
        else:  # or get values from a time object like datetime.time or datetime.date
            self.original_string = None
            self.start_position = None
            self.end_position = None
            self.ordinal = None
            self.number_value = (time_object.hour, time_object.minute) if isinstance(time_object, datetime.time)\
                else (time_object.day, time_object.month, time_object.year)
            self.unit = 'time' if isinstance(time_object, datetime.time) else 'day'
            self.value = time_object

    def __str__(self):
        return str(self.value)

    def __repr__(self):
        return str(self.value)

    def _create_timestamp(self):
        # converts the numerical values of the TimePoint into a time object (either datetime.date or datetime.time)
        if self.unit == 'day':
            if isinstance(self.number_value, int):  # if only one int value is given, assume that it is the day
                today = datetime.date.today()
                return datetime.date(year=today.year, month=today.month, day=self.number_value)
            elif isinstance(self.number_value, tuple):
                if len(self.number_value) == 2:  # if a tuple of two numbers is given, assume that it is (day, month)
                    day, month = self.number_value
                    year = datetime.date.today().year
                    return datetime.date(year=year, month=month, day=day)
                elif len(self.number_value) == 3:  # if a tuple of three numbers is given, assume that it is
                    # (day, month, year)
                    day, month, year = self.number_value
                    if year < 100:  # convert abbreviated years (e.g. 21 instead of 2021) into the full number
                        if year <= ((datetime.date.today().year + 10) - 2000):
                            year = 2000 + year
                        else:
                            year = 1900 + year
                    return datetime.date(year=year, month=month, day=day)
            elif isinstance(self.number_value, str):  # convert time words like 'gestern' into timestamps
                return self._convert_time_words(self.number_value)
        elif self.unit == 'time':
            if isinstance(self.number_value, int):  # if a single int is given, assume that it is hour
                return datetime.time(hour=self.number_value)
            elif isinstance(self.number_value, tuple) and len(self.number_value) == 2:  # with two values,
                # assume that it is (hour, minute)
                hour, minute = self.number_value
                return datetime.time(hour=hour, minute=minute)
        return None

    def _convert_time_words(self, time_word: str):
        # converts time words like 'gestern' or 'letzten Montag' into timestamps, from the reference point of the
        # current date

        # resolve fixed words like 'morgen'
        fixed_intervals = {
            'today': datetime.timedelta(days=0),
            'yesturday': datetime.timedelta(days=-1),
            # 'vorgestern': datetime.timedelta(days=-2),
            'tomorrow': datetime.timedelta(days=1),
            # 'übermorgen': datetime.timedelta(days=2)
        }
        today = datetime.date.today()

        if time_word in fixed_intervals:
            return today + fixed_intervals[time_word]

        # resolve terms that include a reference to a weekday, like 'nächste Woche Mittwoch'
        current_weekday = today.isoweekday()
        for weekday in WEEKDAYS:
            if weekday in time_word:
                numeric_weekday = WEEKDAYS[weekday]
                if 'last' in time_word:
                    weekday_distance = current_weekday - numeric_weekday
                    if weekday_distance < 1:
                        weekday_distance = 7 + weekday_distance  # weekday_distance is now between (1, 7)
                    day =  today - datetime.timedelta(days=weekday_distance)
                    if 'week' in time_word and today.isocalendar().week == day.isocalendar().week:
                        return day - datetime.timedelta(days=7)
                    else:
                        return day
                elif 'next':
                    weekday_distance = numeric_weekday - current_weekday
                    if weekday_distance < 1:
                        weekday_distance = 7 + weekday_distance  # weekday_distance is now between (1, 7)
                    day = today + datetime.timedelta(days=weekday_distance)
                    if 'week' in time_word and today.isocalendar().week == day.isocalendar().week:
                        return day + datetime.timedelta(days=7)
                    else:
                        return day
        return None


class TimeSpan:

    def __init__(self, number_from: Union[Number, TimePoint, None] = None,
                 number_to: Union[Number, TimePoint, None] = None, relation: Optional[str] = None,
                 number_item: Optional[Number] = None):
        # class for time spans like '3 Wochen (lang)' or 'von Montag bis Donnerstag'
        # a time span is defined by its duration (the value) which is converted into hours for all spans
        # initalize by either providing a single number item ('3 Wochen') or two items ('Montag', 'Donnerstag') and
        # their relation ('from_to')
        self.number_item = number_item
        self.number_from = number_from
        self.number_to = number_to
        self.relation = relation
        self.number_value = None if not number_item else number_item.value
        self.original_string = None if not number_item else number_item.original_string

        # if something like 'seit gestern', add a span from yesterday to today (24 hours)
        if self.number_from and not self.number_to and self.relation == 'since':
            if self.number_from.unit == 'day':
                self.number_to = TimePoint(time_object=datetime.date.today())
            elif self.number_from.unit == 'day':
                self.number_to = TimePoint(time_object=datetime.datetime.now().time())
        # if something like 'bis übermorgen', add a span from today to in two days (48 hours)
        elif self.number_to and not self.number_from and self.relation == 'until':
            if self.number_to.unit == 'day':
                self.number_from = TimePoint(time_object=datetime.date.today())
            elif self.number_to.unit == 'day':
                self.number_from = TimePoint(time_object=datetime.datetime.now().time())

        self.value = self._set_value()  # the duration of the span in hours

    def __str__(self):
        return str(self.value)

    def __repr__(self):
        return f'{self.value} hours'

    def _set_value(self):
        # determines the duration of the time span in hours
        if self.number_item:
            unit = self.number_item.unit
            value = self.number_item.value
            if unit == 'minutes':
                return value / 60
            elif unit == 'hours':
                return value
            elif unit == 'days':
                return value * 24
            elif unit == 'weeks':
                return value * 24 * 7
            elif unit == 'months':
                return value * 24 * 30
        if self.number_from and self.number_to:
            # if one of the two time points is not registered as a time point, we will assume that it has the same unit
            # as the other time point
            # example: in 'von 10 bis 12 uhr', the 10 might not be properly identified as 'hour', so we will add it once
            # we processed that there is a span from '10' to '12 uhr'
            if isinstance(self.number_from, TimePoint) and isinstance(self.number_to, Number):
                self.number_to.unit = self.number_from.unit
                if isinstance(self.number_from.number_value, tuple):
                    self.number_to.number_value = (self.number_to.number_value, *self.number_from.number_value[1:])
                self.number_to = TimePoint(self.number_to)
            elif isinstance(self.number_from, Number) and isinstance(self.number_to, TimePoint):
                self.number_from.unit = self.number_to.unit
                if isinstance(self.number_to.number_value, tuple):
                    self.number_from.number_value = (self.number_from.number_value, *self.number_to.number_value[1:])
                self.number_from = TimePoint(self.number_from)
            if isinstance(self.number_from, TimePoint) and isinstance(self.number_to, TimePoint):
                if self.number_from.unit == 'day' and self.number_to.unit == 'day':
                    delta = self.number_to.value - self.number_from.value
                    return delta.days * 24
                elif self.number_from.unit == 'time' and self.number_to.unit == 'time':
                    datetime_from = datetime.datetime.combine(datetime.datetime.today(), self.number_from.value)
                    datetime_to = datetime.datetime.combine(datetime.datetime.today(), self.number_to.value)
                    delta = datetime_to - datetime_from
                    return delta.seconds / (60 * 60)
        return None


class NLU:

    def __init__(self):
        # class for recognizing country and city names as well as time expressions in text
        with open(resource_dir / 'country_synonyms.json', 'r') as f:
            country_synonyms = json.load(f)
            self.countries = {country.lower(): country for country in country_synonyms.keys()}
            self.countries.update({country_syn.lower(): country for country, country_syns in country_synonyms.items()
                                   for country_syn in country_syns})

        with open(resource_dir / 'city_synonyms.json', 'r') as f:
            city_synonyms = json.load(f)
            self.cities = {city.lower(): city for city in city_synonyms.keys() if city != '$REST'}
            self.cities.update({city_syn.lower(): city for city, city_syns in city_synonyms.items()
                                for city_syn in city_syns})

    def extract_boolean(self, user_utterance: str):
        found_booleans = []
        user_utterance = user_utterance.lower()

        # positive values
        if "yes" in user_utterance:
            found_booleans.append(True)
        if "sure" in user_utterance:
            found_booleans.append(True)
        if "absolutely" in user_utterance:
            found_booleans.append(True)
        if "true" in user_utterance:
            found_booleans.append(True)
        if "correct" in user_utterance and not "incorrect" in user_utterance:
            found_booleans.append(True)
        if "right" in user_utterance:
            found_booleans.append(True)
        if "confirm" in user_utterance:
            found_booleans.append(True)
        if "affirmative" in user_utterance:
            found_booleans.append(True)
        
        # negative values
        if "no" in user_utterance:
            found_booleans.append(False)
        if "nope" in user_utterance:
            found_booleans.append(False)
        if "not" in user_utterance:
            found_booleans.append(False)
        if "never" in user_utterance:
            found_booleans.append(False)
        if "incorrect" in user_utterance:
            found_booleans.append(False)
        if "wrong" in user_utterance:
            found_booleans.append(False)
        if "false" in user_utterance:
            found_booleans.append(False)
        if "negative" in user_utterance:
            found_booleans.append(False)
        if "reject" in user_utterance:
            found_booleans.append(False)

        return found_booleans

    def extract_places(self, user_utterance: str):
        # extract cities and countries
        # further questions necessary for: Spanien -> Kanarische Inseln, Frankreich/Paris -> Dep 92, 93 und 94
        user_utterance = user_utterance.lower()

        # recognize countries
        country_string = "|".join([re.escape(country) for country in self.countries.keys()])
        country_regex = re.compile(fr'\b({country_string})\b')
        found_countries = re.findall(country_regex, user_utterance)

        # recognize cities
        city_string = "|".join([re.escape(city) for city in self.cities.keys()])
        city_regex = re.compile(fr'\b({city_string})\b')
        found_cities = re.findall(city_regex, user_utterance)

        return {
            'CITY': [self.cities[city] for city in found_cities],  # list of city names
            'COUNTRY': [self.countries[country] for country in found_countries]  # list of country names
        }

    def extract_time(self, user_utterance: str):
        # extract time points and time spans
        user_utterance = user_utterance.lower()
        found_numbers = self._recognize_numbers(user_utterance)  # numbers like '1', 'eins', 'erster'
        found_time_words = self._recognize_time_words(user_utterance)  # words like 'morgen', 'nächsten Montag'
        all_matches = sorted(found_numbers + found_time_words,
                             key=lambda number: (number.start_position, number.end_position))

        time_points_and_numbers = []
        time_spans = []
        for i in range(len(all_matches)):
            number = all_matches[i]
            # extract units like 'Tage', 'Uhr' and decide whether it is a time point or time span accordingly
            self._extract_time_unit(user_utterance, number)
            if number.time_type:
                if number.time_type == 'point':
                    timepoint = TimePoint(number_item=number)
                    time_points_and_numbers.append(timepoint)
                elif number.time_type == 'span':
                    timespan = TimeSpan(number_item=number)
                    time_spans.append(timespan)
            else:
                time_points_and_numbers.append(number)

        # extract time spans between numbers, like 'vom 13. bis 17. November'
        time_numbers, new_time_spans = self._extract_time_span(user_utterance, time_points_and_numbers)

        # return only their values for easier usage
        time_spans = [time_span.value for time_span in set(time_spans + new_time_spans)]  # time or date objects
        time_points = [number.value for number in time_numbers if isinstance(number, TimePoint)]  # duration in hours
        other_numbers = [number.value for number in time_numbers if number.value not in time_points]  # pure number

        return {
            'time_points': time_points,
            'time_spans': time_spans,
            'other_numbers': other_numbers
        }

    def _recognize_numbers(self, user_utterance: str):
        # recognize numbers given as (a) digits ('1', '19:20', '1.3.2020'), (b) number words ('dreizehn',
        # 'zweitausend'), or (c) ordinal number words ('erster', 'dreizehnte', 'zweitausendstem')

        # (a) digits
        digits_regex = re.compile(r'[0-9]+([.:][0-9]+)*')
        found_digits = [Number(number_string=match.group(0), position_in_utterance=match.span())
                        for match in digits_regex.finditer(user_utterance)]

        # (b) number words
        ones_string = "|".join(NUMBERS['ones'])
        tens_string = "|".join(NUMBERS['tens'])
        hundreds_string = f'(?P<ones_hundred>{ones_string})?hundert'
        thousands_string = f'(?P<ones_thousand>{ones_string})?tausend'
        num_word_regex = re.compile(fr'\b((?P<thousand>{thousands_string})?(?P<hundred>{hundreds_string})?'
                                    fr'((?P<ones_with_tens>{ones_string})?(und)?(?P<tens>{tens_string})|'
                                    fr'(?P<ones>{ones_string}))|(?P<single>{NUMBERS["single"]}))\b')
        found_num_words = [Number(number_string=match.group(0), position_in_utterance=match.span(),
                                  match_dict=match.groupdict()) for match in num_word_regex.finditer(user_utterance)]

        # (c) ordinal number words
        ones_ordinal_string = "|".join(NUMBERS['ones_ordinal'])
        tens_ordinal_string = "|".join(NUMBERS['tens_ordinal'])
        hundreds_ordinal_string = f'(?P<ones_hundred>{ones_string})?hundreth(r|n|m)?'
        thousands_ordinal_string = f'(?P<ones_thousand>{ones_string})?thousandth(r|n|m)?'
        num_word_ordinal_regex = re.compile(fr'\b((?P<thousand>{thousands_ordinal_string})|'
                                            fr'(?P<hundred>{hundreds_ordinal_string})|'
                                            fr'(?P<ones_with_tens>{ones_string})?(und)?(?P<tens>{tens_ordinal_string})|'
                                            fr'(?P<ones>{ones_ordinal_string})|(?P<single>{NUMBERS["single"]}))\b')
        found_num_ordinal_words = [Number(number_string=match.group(0), position_in_utterance=match.span(),
                                          match_dict=match.groupdict(), ordinal=True)
                                   for match in num_word_ordinal_regex.finditer(user_utterance)]

        return found_digits + found_num_words + found_num_ordinal_words

    def _recognize_time_words(self, user_utterance: str):
        # recognize time words that are in relation to the current day, like 'heute', 'vorgestern', 'letzten Samstag'
        weekdays_string = "|".join(WEEKDAYS.keys())
        time_word_regex = re.compile(fr'(today|yesturday|tomorrow|'
                                     fr'(last|next|this)\s*({weekdays_string})|'
                                     fr'|(last|next|this)\s*week\s*({weekdays_string}))')
        return [Number(number_string=match.group(0), position_in_utterance=match.span())
                for match in time_word_regex.finditer(user_utterance) if match.group(0)]


    def _extract_time_unit(self, user_utterance: str, number: Number):
        # recognize the unit of the number; e.g. '3 Tage' means a duration over three days but '3 Uhr' means a time
        # point at a certain hour
        # also recognize if a month is given with its name, e.g '3. März'
        months_string = "|".join(MONTHS.keys())
        unit_regex = re.compile(fr'{number.original_string}\s*((?P<hours>(hours?|hrs))|(?P<days>days?)|(?P<weeks>weeks?)|(?P<months>months?)|'
                                fr"(?P<time>(o'clock|a.m.|p.m.|am|pm))|(?P<month>{months_string}))")
        match = re.search(unit_regex, user_utterance)
        if match:
            for unit, found_string in match.groupdict().items():
                if found_string:
                    if unit == 'month':
                        number.add_month(found_string)
                    else:
                        number.add_unit(unit)

    def _extract_time_span(self, user_utterance: str, time_numbers: List[Union[Number, TimePoint]]):
        # extract a time span either (a) between the current day and a recognized day, or (b) between two timepoints
        time_spans = []
        until_spans = {}

        # (a) time span between the current day and the recognized day (e.g. 'seit gestern', 'bis morgen')
        for time_match in time_numbers:
            match_string = time_match.original_string
            if not match_string:
                continue
            single_time_regex = re.compile(fr'((?P<since>since\s*(the\s*)?{match_string})|'
                                           fr'(?P<until>until\s*(the\s*)?{match_string}))')
            match = re.search(single_time_regex, user_utterance)
            if match:
                for label, found_string in match.groupdict().items():
                    if found_string:
                        time_match.time_type = 'span'
                        if label == 'since':
                            time_spans.append(TimeSpan(number_from=time_match, number_to=None, relation='since'))
                        elif label == 'until':
                            until_spans[time_match] = TimeSpan(number_from=None, number_to=time_match, relation='until')

        remove_time_numbers = []
        add_time_numbers = []
        # (b) time span between two time points
        # might also just extend time points that consist of several numbers (like '1 Uhr 30' or '1. Juni 1987')
        if len(time_numbers) > 1:
            months_string = "|".join(MONTHS.keys())
            for match_1, match_2 in list(zip(time_numbers[:-1], time_numbers[1:])):
                string_1, string_2 = match_1.original_string, match_2.original_string
                if not string_1 or not string_2:
                    continue
                multi_time_regex = re.compile(fr'((?P<from_to>(from)?\s*{string_1}\s*until (the)?\s*{string_2})|'
                                              fr'(?P<from_to_hyphen>{string_1}\s*-\s*{string_2})|'
                                              fr'(?P<between>between\s*{string_1}\s*and\s*{string_2})|'
                                            #   fr"(?P<time>{string_1}\s*uhr\s*{string_2})|"
                                              fr'(?P<month>{string_1}\.?\s*({months_string})\s*{string_2}))')
                match = re.search(multi_time_regex, user_utterance)
                if match:
                    for label, found_string in match.groupdict().items():
                        if found_string:
                            if label == 'from_to' and match_2 in until_spans:
                                # in 'von 1 bis 3', the 'bis 3' is not in reference to the current day (see (a))
                                del until_spans[match_2]
                            if label == 'from_to_hyphen':  # '1-3' should be treated like 'von 1 bis 3'
                                label = 'from_to'
                            if label == 'time':
                                # for time points it is assumed that the first number is the hour and the second the
                                # minute
                                if isinstance(match_1.number_value, int) and isinstance(match_2.number_value, int):
                                    if isinstance(match_1, TimePoint) and match_1.unit == 'time':
                                        # extend an existing time point
                                        match_1.value = match_1.value.replace(minute=match_2.number_value)
                                        match_1.number_value = (match_1.number_value, match_2.number_value)
                                        remove_time_numbers.append(match_2)
                                    else:
                                        # create a new time point
                                        add_time_numbers.append(TimePoint(time_object=datetime.time(
                                            hour=match_1.number_value, minute=match_2.number_value)))
                                        remove_time_numbers.extend([match_1, match_2])
                            elif label == 'month':
                                # for months, it is assumed that the first number is the day, the second the year,
                                # and the month is given as word in between
                                if isinstance(match_2.number_value, int):
                                    found_month = [index for month, index in MONTHS.items()
                                                   if month in found_string][0]
                                    if isinstance(match_1, TimePoint) and match_1.unit == 'day':
                                        # extend an existing time point
                                        if isinstance(match_1.number_value, int):
                                            # add month and year
                                            match_1.value = match_1.value.replace(month=found_month,
                                                                                  year=match_2.number_value)
                                            match_1.number_value = (match_1.number_value, found_month,
                                                                    match_2.number_value)
                                            remove_time_numbers.append(match_2)
                                        elif isinstance(match_1.number_value, tuple) and len(match_1.number_value) == 2:
                                            # add year
                                            match_2.value = match_2.value.replace(year=match_2.number_value)
                                            match_1.number_value = (*match_1.number_value, match_2.number_value)
                                            remove_time_numbers.append(match_2)
                                    else:
                                        # create a new time point
                                        add_time_numbers.append(TimePoint(time_object=datetime.date(
                                            year=match_2.number_value, month=found_month, day=match_1.number_value)))
                                        remove_time_numbers.extend([match_1, match_2])
                            else:
                                # create a time span between two time points
                                time_span = TimeSpan(number_from=match_1, number_to=match_2, relation=label)
                                # check whether the time points have been changed during the time span creation
                                if time_span.number_from != match_1:
                                    add_time_numbers.append(time_span.number_from)
                                    remove_time_numbers.append(match_1)
                                if time_span.number_to != match_2:
                                    add_time_numbers.append(time_span.number_to)
                                    remove_time_numbers.append(match_2)
                                time_spans.append(time_span)
        time_spans.extend(until_spans.values())
        time_numbers.extend(add_time_numbers)
        time_numbers = list(set(time_numbers) - set(remove_time_numbers))
        return time_numbers, time_spans

# if __name__ == '__main__':
#     # examples, should be recognized correctly (probably incomplete)
#     nlu = NLU()
#     print(nlu.extract_places('Ich war in deutschland, Österreich, und in der Schweiz.'))
#     print(nlu.extract_time('Es war am 19.05. um 10 Uhr'))
#     print(nlu.extract_time('Die Konferenz dauerte siebzehn Tage.'))
#     #print(nlu.extract_time('Es waren zwischen sieben und achtzehn Tage'))
#     print(nlu.extract_time('Vom ersten bis fünften März'))
#     print(nlu.extract_time('Es fand am 10. Oktober 2020 statt'))
#     print(nlu.extract_time('Ich glaube sie startete um 10 Uhr 34, wenn ich mich nicht irre'))
#     print(nlu.extract_time('Oder war es doch um 10:35?'))
#     print(nlu.extract_time('Ich bin seit dem 5.10.21 auf der Tagung.'))
#     print(nlu.extract_time('Ich denke es war um 10.30 Uhr'))
#     print(nlu.extract_time('Der Workshop war 10-12 Uhr.'))
