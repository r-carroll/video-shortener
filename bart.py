from transformers import BartTokenizer, BartForConditionalGeneration

def generate_chapter_titles(input_text):
    # Load BART model and tokenizer
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
    model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")

    # Tokenize the input text
    inputs = tokenizer(input_text, return_tensors="pt", max_length=1024, truncation=True)

    # Generate a summary for the entire document
    summary_ids = model.generate(**inputs, max_length=150, min_length=50, length_penalty=2.0, num_beams=4, early_stopping=True)
    document_summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    # Split the document into sections based on paragraphs or other criteria
    sections = input_text.split("\n\n")  # You might need to adjust this based on the document structure

    # Generate summaries for each section and use them as proposed chapter titles
    proposed_chapters = []
    for i, section in enumerate(sections):
        section_inputs = tokenizer(section, return_tensors="pt", max_length=1024, truncation=True)
        section_summary_ids = model.generate(**section_inputs, max_length=150, min_length=50, length_penalty=2.0, num_beams=4, early_stopping=True)
        section_summary = tokenizer.decode(section_summary_ids[0], skip_special_tokens=True)
        proposed_chapters.append({"chapter_number": i + 1, "chapter_title": section_summary})

    return proposed_chapters

# Example usage
input_text = """
Mr. President and Gentlemen of the Convention.

If we could first know where we are, and whither we are tending, we could better judge what to do, and how to do it.

We are now far into the fifth year, since a policy was initiated, with the avowed object, and confident promise, of putting an end to slavery agitation.

Under the operation of that policy, that agitation has not only, not ceased, but has constantly augmented.

In my opinion, it will not cease, until a crisis shall have been reached, and passed -

"A house divided against itself cannot stand."

I believe this government cannot endure, permanently half slave and half free.

I do not expect the Union to be dissolved - I do not expect the house to fall - but I do expect it will cease to be divided.

It will become all one thing, or all the other.

Either the opponents of slavery, will arrest the further spread of it, and place it where the public mind shall rest in the belief that it is in course of ultimate extinction; or its advocates will push it forward till it shall become alike lawful in all the States, old as well as new-North as well as South.

Have we no tendency to the latter condition?

Let any one who doubts, carefully contemplate that now almost complete legal combination - piece of machinery so to speak- compounded of the Nebraska doctrine, and the Dred Scott decision. Let him consider not only what work the machinery is adapted to do, and how well adapted; but also, let him study the history of its construction, and trace, if he can, or rather fail, if he can, to trace the evidences of design and concert of action, among its chief bosses, from the beginning.

But, so far, Congress only, had acted; and an indorsement by the people, real or apparent, was indispensable, to save the point already gained, and give chance for more.

The new year of 1854 found slavery excluded from more than half the State by State Constitutions, and from most of the national territory by congressional prohibition.

Four days later, commenced the struggle, which ended in repealing that congressional prohibition.

This opened all the national territory to slavery; and was the first point gained.

This necessity had not been overlooked; but had been provided for, as well as might be, in the notable argument of "squatter sovereignty," otherwise called "sacred right of self government," which latter phrase, though expressive of the only rightful basis of any government, was so perverted in this attempted use of it as to amount to just this: That if any one man, choose to enslave another, no third man shall be allowed to object.

That argument was incorporated into the Nebraska bill itself, in the language which follows: "It being the true intent and meaning of this act not to legislate slavery into any Territory or State, nor to exclude it therefrom; but to leave the people thereof perfectly free to form and regulate their domestic institutions in their own way, subject only to the Constitution of the United States."

Then opened the roar of loose declamation in favor of "Squatter Sovereignty," and "Sacred right of self government."

"But," said opposition members, "let us be more specific- let us amend the bill so as to expressly declare that the people of the Territory may exclude slavery." "Not we," said the friends of the measure; and down they voted the amendment.

While the Nebraska bill was passing through congress, a law case, involving the question of a negro's freedom, by reason of his owner having voluntarily taken him first into a free State and then a territory covered by the congressional prohibition, and held him as a slave for a long time in each, was passing through the U.S. Circuit Court for the District of Missouri; and both Nebraska bill and law suit were brought to a decision in the same month of May, 1854. The negro's name was "Dred Scott," which name now designates the decision finally made in the case.

Before the then next Presidential election, the law case came to, and was argued in the Supreme Court of the United States; but the decision of it was deferred until after the election. Still, before the election, Senator Trumbull, on the floor of the Senate, requests the leading advocate of the Nebraska bill to state his opinion whether the people of a territory can constitutionally exclude slavery from their limits; and the latter answers, "That is a question for the Supreme Court."

The election came. Mr. Buchanan was elected, and the indorsement, such as it was, secured. That was the second point gained. The indorsement, however, fell short of a clear popular majority by nearly four hundred thousand votes, and so, perhaps, was not over-whelmingly reliable and satisfactory.

The outgoing President, in his last annual message, as impressively as possible echoed back upon the people the weight and authority of the indorsement.

The Supreme Court met again, did not announce their decision, but ordered a re-argument.

The Presidential inauguration came, and still no decision of the court; but the incoming President, in his inaugural address, fervently exhorted the people to abide by the forthcoming decision, whatever it might be.

Then, in a few days, came the decision.

The reputed author of the Nebraska bill finds an early occasion to make a speech at this capitol indorsing the Dred Scott Decision, and vehemently denouncing all opposition to it.

The new President, too, seizes the early occasion of the Silliman letter to indorse and strongly construe that decision, and to express his astonishment than any different view had ever been entertained.

At length a squabble springs up between the President and the author of the Nebraska bill, on the mere question of fact, whether the Lecompton constitution was or was not, in any just sense, made by the people of Kansas; and in that quarrel the latter declares that all he wants is a fair vote for the people, and that he cares not whether slavery be voted down or voted up. I do not understand his declaration that he cares not whether slavery be voted down or voted up, to be intended by him other than as an apt definition of the policy he would impress upon the public mind - the principle for which he declares he has suffered much, and is ready to suffer to the end.

And well may he cling to that principle. If he has any parental feeling, well may he cling to it. That principle, is the only shred left of his original Nebraska doctrine. Under the Dred Scott decision, "squatter sovereignty" squatted out of existence, tumbled down like temporary scaffolding - like the mold at the foundry served through one blast and fell back into loose sand - helped to carry an election, and then was kicked to the winds. His late joint struggle with the Republicans, against the Lecompton Constitution, involves nothing of the original Nebraska doctrine. That struggle was made on a point, the right of a people to make their own constitution, upon which he and the Republicans have never differed.

The several points of the Dred Scott decision, in connection with Senator Douglas' "care not" policy, constitute the piece of machinery, in its present state of advancement.

The working points of that machinery are:

First, that no negro slave, imported as such from Africa, and no descendant of such slave can ever be a citizen of any State, in the sense of that term as used in the Constitution of the United States.

This point is made in order to deprive the negro, in every possible event, of the benefit of that provision of the United States Constitution, which declares that -

"The citizens of each State shall be entitled to all privileges and immunities of citizens in the several States."

Secondly, that "subject to the Constitution of the United States," neither Congress nor a Territorial Legislature can exclude slavery from any United States Territory.

This point is made in order that individual men may fill up the territories with slaves, without danger of losing them as property, and thus enhance the chances of permanency to the institution through all the future.

Thirdly, that whether the holding a negro in actual slavery in a free State, makes him free, as against the holder, the United States courts will not decide, but will leave to be decided by the courts of any slave State the negro may be forced into by the master.

This point is made, not to be pressed immediately; but, if acquiesced in for a while, and apparently indorsed by the people at an election, then ro sustain the logical conclusion that what Dred Scott's master might lawfully do with Dred Scott, in the free State of Illinois, every other master may lawfully do with any other one or one thousand slaves, in Illinois, or in any other free State.

Auxiliary to all this, and working hand in hand with it, the Nebraska doctrine, or what is left of it, is to educate and mould public opinion, at least Northern public opinion, to not care whether slavery is voted down or voted up.

This shows exactly where we now are; and partially also, whither we are tending.

It will throw additional light on the latter, to go back, and run the mind over the string of historical facts already stated. Several things will now appear less dark and mysterious than they did when they were transpiring. The people were to be left "perfectly free" "subject only to the Constitution." What the Constitution had to do with it, outsides could not then see. Plainly enough now, it was an exactly fitted nitch for the Dred Scott decision to afterward come in, and declare that perfect freedom of the people, to be just no freedom at all.

Why was the amendment, expressly declaring the right of the people to exclude slavery, voted down? Plainly enough now, the adoption of it, would have spoiled the nitch for the Dred Scott decision.

Why was the court decision held up? Why, even a Senator's individual opinion withheld, till after the Presidential election? Plainly enough now, the speaking out then would have damaged the "perfectly free" argument upon which the election was to be carried.

Why the outgoing President's felicitation on the indorsement? Why the delay of a reargument? Why the incoming President's advance exhortation in favor of the decision?

These things look like the cautious patting and petting of a spirited horse, preparatory to mounting him, when it is dreaded that he may give the rider a fall.

Any why the hasty after indorsements of the decision by the President and others?

We cannot absolutely know that all these exact adaptations are the result of preconcert. But when we see a lot of framed timbers, different potions of which we know have been gotten out at different times and places and by different workmen,- Stephen, Franklin, Roger and James, for instance-and we see these timbers joined together, and see they exactly make the frame of a house or a mill, all the tenons and mortieses exactly fitting, and all the lengths and proportions of the different pieces exactly adapted to their respective places, and not a piece too many or too few-not omitting even scaffolding-or, if a single piece be lacking, we see the place in the frame exactly fitted and prepared to yet bring such piece in-in such a case, we find it impossible not to believe that Stephen and Franklin and Roger and James all understood one another from the beginning, and all worked upon a common plan or draft drawn up before the first lick was struck.

It should not be overlooked that, by the Nebraska bill, the people of State as well as Territory, were to be left "perfectly free" "subject only to the Constitution."

Why mention a State? They were legislating for territories, and not for or about States. Certainly the people of a State are and ought to be subject to the Constitution of the United States; but why is mention of this lugged into this merely territorial law? Why are the people of a territory and the people of a state therein lumped together, and their relation to the Constitution therein treated as being precisely the same?

While the opinion of the Court, by Chief Justice Taney, in the Dred Scott case, and the separate opinions of all the concurring Judges, expressly declare that the Constitution of the United States neither permits Congress nor a territorial legislature to exclude slavery from any United States territory, they all omit to declare whether or not the same Constitution permits a state, or the people of a State to exclude it.

Possibly, this is a mere omission; but who can be quite sure, if McLean or Curtis had sought to get into the opinion a declaration of unlimited power in the people of a state to exclude slavery from their limits, just as Chase and Mace sought to get such declaration, in behalf of the people of a territory, into the Nebraska bill-I ask, who can be quite sure that it would not have been voted down, in the one case, as it had been in the other?

The nearest approach to the point of declaring the power of a State over slavery, is made by Judge Nelson. He approaches it more than once, using the precise idea, and almost the language too, of the Nebraska act. On one occasion his exact language is, "except in cases where the power is restrained by the Constitution of the United States, the law of the State is supreme over the subject of slavery within its jurisdiction."

In what cases the power of the states is so restrained by the U.S. Constitution is left an open question, precisely as the same question, as to the restraint on the power of the territories was left open in the Nebraska act. Put that and that together, and we have another nice little nitch, which we may, ere long, see filled with another Supreme Court decision, declaring that the Constitution of the United States does not permit a state to exclude slavery from its limits.

And this may be expected if the doctrine of "care not whether slavery be voted down or voted up," shall gain upon the public mind sufficiently to give promise that such a decision can be maintained when made.

Such a decision is all that slavery now lacks of being alike lawful in all the States.

Welcome or unwelcome, such decision is probably coming, and will soon be upon us, unless the power of the present political dynasty shall be met and overthrown.

We shall lie down pleasantly dreaming that the people of Missouri are on the verge of making their State free; and we shall awake to the reality, instead, that the Supreme Court has made Illinois a slave State.

To meet and overthrow the power of that dynasty, is the work now before all those who would prevent that consummation.

That is what we have to do.

But how can we best do it?

There are those who denounce us openly to their own friends, and yet whisper us softly, that Senator Douglas is the aptest instrument there is, with which to effect that object. They do not tell us, nor has he told us, that he wishes any such object to be effected. They wish us to infer all, from the facts, that he now has a little quarrel with the present head of the dynasty; and that he has regularly voted with us, on a single point, upon which, he and we, have never differed.

They remind us that he is a great man, and that the largest of us are very small ones. Let this be granted. But "a living dog is better than a dead lion." Judge Douglas, if not a dead lion for this work, is at least a caged and toothless one. How can he oppose the advance of slavery? He don't care anything about it. His avowed mission is impressing the "public heart" to care nothing about it.

A leading Douglas Democratic newspaper thinks Douglas' superior talent will be needed to resist the revival of the African slave trade.

Does Douglas believe an effort to revive that trade is approaching? He has not said so. Does he really think so? But if it is, how can he resist it? For years he has labored to prove it a sacred right of white men to take negro slaves into the new territories. Can he possibly show that it is less a sacred right to buy them where they can be brought cheapest? And, unquestionably they can be bought cheaper in Africa than in Virginia.

He has done all in his power to reduce the whole question of slavery to one of a mere right of property; and as such, how can he oppose the foreign slave trade-how can he refuse that trade in that "property" shall be "perfectly free"-unless he does it as a protection to the home production? And as the home producers will probably not ask the protection, he will be wholly without a ground of opposition.

Senator Douglas holds, we know, that a man may rightfully be wiser today than he was yesterday-that he may rightfully change when he finds himself wrong.

But, can we for that reason, run ahead, and infer that he will make any particular change, of which he, himself, has given no intimation? Can we safely base our action upon any such vague inference?

Now, as ever, I wish to not misrepresent Judge Doulgas' position, question his motives, or do aught that can be personally offensive to him.

Whenever, if ever, he and we can come together on principle so that our great cause may have assistance from his great ability, I hope to have interposed no adventitious obstacle.

But clearly, he is not now with us-he does not pretend to be-he does not promise to ever be.

Our cause, then, must be intrusted to, and conducted by its own undoubted friends-those whose hands are free, whose hearts are in the work-who do care for the result.

Two years ago the Republicans of the nation mustered over thirteen hundred thousand strong.

We did this under the single impulse of resistance to a common danger, with every external circumstance against us.

Of strange, discordant, and even, hostile elements, we gathered from the four winds, and formed and fought the battle through, under the constant hot fire of a disciplined, proud, and pampered enemy.

Did we brave all then to falter now? - now - when that same enemy is wavering, dissevered, and belligerent?

This result is not doubtful. We shall not fail-if we stand firm, we shall not fail.
"""

result = generate_chapter_titles(input_text)
for chapter in result:
    print(f"Chapter {chapter['chapter_number']} Title: {chapter['chapter_title']}")
